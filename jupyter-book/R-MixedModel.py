# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
library(reticulate)
library(ggpubr)
library(lme4)
library(rstatix)
library(glmmTMB)
library(DHARMa)
library(modelsummary)
library(stargazer)
library(repr)
library(dplyr)
library(xtable)
library(texreg)
library(geepack)
library(MCMCglmm)
library(parallel)
library(coda)
library(effects)
library(ggeffects)
library(gridExtra)

# %%
options(repr.plot.width = 20, repr.plot.height = 10)

# %%
llm.coop <- import("llm_cooperation.main")

# %%
config=llm.coop$Configuration(
    grid=dict(
        temperature=c(0.1, 0.6),
        model=c("gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"),
        max_tokens=list(500)
    ),
    experiment_names=c("dilemma", "dictator"),
    num_participant_samples=30,
    num_replications=3
)

# %%
results <- llm.coop$load_all(config)
results

# %%
ggboxplot(results, x="Participant_group", y = "Cooperation frequency", color="Partner Condition")

# %%
names(results)

# %%
names(results)[1] = "Participant_condition"
names(results)[2] = "Partner_condition"
names(results)[4] = "Cooperation_frequency"

# %%
results.clean <- results %>% 
    convert_as_factor(Experiment, Participant_group, Partner_condition, t, Model, Participant_id, Participant_prompt_index, 
                      Participant_chain_of_thought, Participant_label, Participant_case, Participant_pronoun, 
                      Participant_defect_first, Participant_labels_reversed) %>% 
    filter(!is.na(Cooperation_frequency))

# %%
results.clean

# %%
levels(results.clean$Participant_pronoun)

# %%
levels(results.clean$Experiment)

# %%
levels(results.clean$Model)

# %%
levels(results.clean$Participant_group)

# %%
levels(results.clean$Partner_condition)

# %%
levels(results.clean$Participant_label)

# %%
levels(results.clean$Participant_case)

# %%
results.clean$Participant_group <- relevel(results.clean$Participant_group, ref='Control')
results.clean$Partner_condition <- relevel(results.clean$Partner_condition, ref='tit for tat D')
results.clean$Model <- relevel(results.clean$Model, ref='gpt-3.5-turbo-0613')
results.clean$Participant_pronoun <- relevel(results.clean$Participant_pronoun, ref='they')
results.clean$Participant_case <- relevel(results.clean$Participant_case, ref='standard')
results.clean$Particpant_label <- relevel(results.clean$Participant_label, ref='colors')

# %%
names(results.clean)

# %%
results.clean <- results.clean %>% select(-Choices, -Transcript, -Participant_condition)

# %%
names(results.clean)

# %%
hist(results.clean$Cooperation_frequency)

# %%
stargazer(results.clean)

# %%
summary.stats <- results.clean %>%  
  summarise(
    .by = c(Experiment, Participant_group),
    Mean = mean(Cooperation_frequency, na.rm = TRUE),
    SD = sd(Cooperation_frequency, na.rm = TRUE),
    Median = median(Cooperation_frequency, na.rm = TRUE),
    IQR = IQR(Cooperation_frequency, na.rm = TRUE),
    N = n()
  )
summary.stats

# %%
latex_table <- xtable(summary.stats)
print(latex_table, type = "latex", include.rownames = TRUE)

# %%
results.grouped <- results.clean %>% group_by(Experiment, Participant_group, Partner_condition, t, Model, Temperature)
results.grouped

# %%
options(repr.plot.height=24)

ggplot(results.clean, aes(x = Participant_group, y = Cooperation_frequency)) +
  geom_boxplot() +
  facet_wrap(. ~ Model + Temperature + Participant_labels_reversed + Partner_condition, ncol=6)

# %%
results.pd = results.clean[results.clean$Experiment == 'dilemma', ]
results.pd

# %%
results.pd$Num_cooperates = round(results.pd$Cooperation_frequency * 6)

# %%
num_defects <- 6 - results.pd$Num_cooperates
summary(num_defects)

# %%
verbatim <- function(x)
{
  txt <- trimws(capture.output(x))
  txt <- c("\\begin{verbatim}", txt, "\\end{verbatim}")
  cat(txt, sep="\n")
}

# %%
model.pd <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
                 Participant_group + Participant_group:Partner_condition + t + Model + Temperature +
                 (1|Participant_id),
               data = results.pd,
               family = betabinomial)
summary(model.pd)

# %%
summary(model.pd)

# %%
model.pd.1 <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
                 Participant_group + Participant_group:Partner_condition + t + Model + Temperature +
                 Participant_group:Partner_condition:Model +
                 (1|Participant_id),
               data = results.pd,
               family = betabinomial)
summary(model.pd.1)

# %%
texreg(model.pd, caption="Fitted model for Prisoners Dilemma", label="table:pd-estimates", file="pd-estimates.tex", single.row=TRUE, fontsize="small")

# %%
xtable(lme4::formatVC(summary(model.pd)$varcor$cond))

# %%
m <- coef(summary(model.pd.1))$cond
m[, 1] <- exp(m[, 1])
m[, 2] <- exp(m[, 2])
colnames(m)[1] <- 'Estimate (Odds ratio)'
m

# %%
m[m[, 4] <= 0.05, ]

# %%

# %%
xtable(coef(summary(model.pd))$cond, digits=3)

# %%
predicted.plot <- ggpredict(model.pd, c("Partner_condition", "Participant_group"))
plot(predicted.plot)

# %%
predicted.plot <- ggpredict(model.pd, c("Partner_condition", "Participant_group", "Model [gpt-3.5-turbo-1106]"))
plot(predicted.plot)

# %%
predicted.plot <- ggpredict(model.pd, c("Partner_condition", "Participant_group", "Model"))
plot(predicted.plot)

# %%
pdf("figs/glmm-predicted.pdf", width=24, height=12)
plot(predicted.plot)
dev.off()

# %%
interaction.plot <- function(model, participant.group) {
    model.term <- sprintf("Model [%s]", model)
    participant.group.term <- sprintf("Participant_group [%s]", participant.group)
    
    predicted <- ggpredict(model.pd, c("Partner_condition", model.term, participant.group.term))

    p <- plot(predicted) + geom_line(aes(x = x, y = predicted), color = "blue") + 
      geom_point(aes(x = x, y = predicted), color = "red") +
      labs(title=participant.group)

    # pdf(sprintf("figs/glmm-interaction-%s-%s.pdf", model,  participant.group))
    # print(p)
    # dev.off()
    
    return(p)
}

interaction.plots <- function(model) {
    do.call(grid.arrange, 
            c(top=model, mclapply(
                levels(results.pd$Participant_group), 
                function(g) interaction.plot(model, g), 
                mc.cores=6
            ))
    )
}

pdf.interaction.plots <- function(model) {
    pdf(sprintf("figs/interaction-plots-%s", model))
    result <- interaction.plots(model)
    print(result)
    dev.off()
    return(result)
}

all.interaction.plots <- function() {
    lapply(levels(results.pd$Model), pdf.interaction.plots)
}

# %%
all.interaction.plots()

# %%
predicted <- ggpredict(model.pd, c("Participant_group", "Model [gpt-3.5-turbo-1106]", "Partner_condition [unconditional cooperate]"))

# Plot with ggeffects and ggplot2
p <- plot(predicted)

# Modify the plot to use geom_line for straight lines
p + geom_line(aes(x = x, y = predicted), color = "blue") + 
  geom_point(aes(x = x, y = predicted), color = "red")  # Add points for actual data

# %%

# %%
plot(ggpredict(model.pd, c("Participant_group", "Model")))

# %%
plot(ggpredict(model.pd, c("Partner_condition", "Model")))

# %%
predicted <- ggpredict(model.pd.1, c("Participant_group", "Model", "Partner_condition"))
predicted

# %%
plot(predicted)

# %%
predicted <- ggpredict(model.pd.1, "Temperature")
predicted

# %%
#effects.1 <- allEffects(model.pd.1)
#plot(effects.1)

# %%
plot(allEffects(model.pd))

# %%
options(repr.plot.height=12)
simulationOutput <- simulateResiduals(fittedModel = model.pd, plot = TRUE, integerResponse=TRUE)

# %%
options(repr.plot.height=10)
pdf("residuals-plots.pdf")
simulationOutput <- simulateResiduals(fittedModel = model.pd, plot = TRUE, integerResponse=TRUE)
dev.off()

# %%
pdf("residuals-hist.pdf")
hist(residuals(simulationOutput))
dev.off()

# %%
# Fit a GEE model with an interaction term
gee_model <- geeglm(cbind(Num_cooperates, 6 - Num_cooperates) ~ Participant_group + Partner_condition + t + Model + Temperature +
                 Partner_condition:Model + Participant_group:Model, 
                    id = Participant_id, 
                    data = results.pd, 
                    family = binomial, 
                    corstr = "exchangeable")

# %%
summary(gee_model)

# %%
residuals_gee <- residuals(gee_model, type = "pearson")
# Plotting Pearson residuals against fitted values
fitted_values <- fitted(gee_model)
plot(fitted_values, residuals_gee, xlab = "Fitted Values", ylab = "Pearson Residuals")
abline(h = 0, col = "red")


# %%
# Overdispersion check
overdispersion_check <- sum(residuals_gee^2) / gee_model$df.residual
overdispersion_check

# %%
#model.pd.factorial <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
#                 Participant_group + Partner_condition + t + Model + Temperature +
#                 Partner_condition:Model + Participant_group:Model + Participant_labels_reversed:Participant_label + Participant_labels_reversed:Model +
#                 Participant_label + Participant_chain_of_thought + Participant_pronoun + Participant_defect_first + Participant_labels_reversed,
#               data = results.clean,
#               family = betabinomial)
#summary(model.pd.factorial)

# %%
simulationOutput.factorial<- simulateResiduals(fittedModel = model.factorial, plot = TRUE, integerResponse=TRUE)

# %%
hist(residuals(simulationOutput.factorial))

# %%
plotResiduals(simulationOutput.factorial, results.clean$Participant_group)

# %% [markdown]
# ## Bayesian MCMC GLMM

# %%
# Set up a list for priors
prior <- list(
    R = list(V = 1, nu = 0.002), # Prior for the residual covariance matrix
    G = list(G1 = list(V = 1, nu = 0.002)) # Prior for the random effects
)

estimate_glmm <- function() {
# Fit the model with the specified priors
    MCMCglmm(fixed=Num_cooperates ~ Participant_group + Partner_condition + t + Model + Model:Partner_condition:Participant_group, 
                          random=~ us(1):Participant_id,
                          prior = prior,
                          data = results.pd,
                          verbose = FALSE)
}

model.pd.mcmc <- estimate_glmm()

# %%
plot(model.pd.mcmc$Sol)

# %%
plot(model.pd.mcmc$VCV)

# %%
m6 <- mclapply(1:12, function(i) { estimate_glmm() }, mc.cores=12)
m6 <- lapply(m6, function(m) m$Sol)
m6 <- do.call(mcmc.list, m6)

# %%
par(mfrow=c(4,2), mar=c(2,2,1,2))
gelman.plot(m6, auto.layout=F)

# %%
gelman.diag(m6)

# %%
par(mfrow=c(8,2), mar=c(2, 1, 1, 1))
plot(m6, ask=F, auto.layout=F)

# %%
summary(m6)

# %%
plot.estimates <- function(x) {
  if (class(x) != "summary.mcmc")
    x <- summary(x)
  n <- dim(x$statistics)[1]
  par(mar=c(2, 7, 4, 1))
  plot(x$statistics[,1], n:1,
       yaxt="n", ylab="",
       xlim=range(x$quantiles)*1.2,
       pch=19,
       main="Posterior means and 95% credible intervals")
  grid()
  axis(2, at=n:1, rownames(x$statistics), las=2)
  arrows(x$quantiles[,1], n:1, x$quantiles[,5], n:1, code=0)
  abline(v=0, lty=2)
}

options(repr.plot.height=24, repr.plot.width=20)
plot.estimates(m6)
