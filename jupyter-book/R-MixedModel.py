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
libs <- c('ggpubr',
'reticulate',
'lme4',
'rstatix',
'glmmTMB',
'DHARMa',
'modelsummary',
'stargazer',
'repr',
'dplyr',
'xtable',
'texreg',
'geepack',
'MCMCglmm',
'parallel',
'coda',
'effects',
'ggeffects',
'gridExtra',
'purrr',
'memoise',
'R.cache',
'patchwork')

lapply(libs, function(lib) library(lib, character.only=TRUE))

# %%
citations <- lapply(libs, function(lib) print(citation(lib), style="Bibtex"))

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
levels(results.clean$Partner_condition) <- c('T4TC', 'T4TD', 'C', 'D')

# %%
levels(results.clean$Participant_label)

# %%
levels(results.clean$Participant_case)

# %%
results.clean$Participant_group <- relevel(results.clean$Participant_group, ref='Control')
results.clean$Partner_condition <- relevel(results.clean$Partner_condition, ref='D')
results.clean$Model <- relevel(results.clean$Model, ref='gpt-3.5-turbo-0613')
results.clean$Participant_pronoun <- relevel(results.clean$Participant_pronoun, ref='they')
results.clean$Participant_case <- relevel(results.clean$Participant_case, ref='standard')
results.clean$Particpant_label <- relevel(results.clean$Participant_label, ref='colors')

# %%
levels(results.clean$Partner_condition)

# %%
results.clean$Partner_condition <- 
    factor(results.clean$Partner_condition, 
           levels = c('D', 'T4TD', 'T4TC', 'C'))

# %%
levels(results.clean$Partner_condition)

# %%
results.clean$Participant_group <- 
    factor(results.clean$Participant_group, 
           levels = c('Control', 'Selfish', 'Competitive', 'Cooperative', 'Altruistic'))

# %%
levels(results.clean$Participant_group)

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
results.dictator <- results.clean[results.clean$Experiment == 'dictator', ]
results.dictator

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
results.dictator %>% count(Cooperation_frequency == 1) %>% mutate(prop = n / sum(n))

# %%
results.dictator %>% count(Cooperation_frequency == 0) %>% mutate(prop = n / sum(n))

# %%
results.dictator <- results.dictator %>% 
    mutate(Cooperation_frequency = ifelse(Cooperation_frequency == 0, 0.001, Cooperation_frequency)) %>%
    mutate(Cooperation_frequency = ifelse(Cooperation_frequency == 1, 0.999, Cooperation_frequency))

# %%
model.dictator <- glmmTMB(Cooperation_frequency  ~
                 Participant_group + Participant_group:Model + t + Model + Temperature +
                 (1|Participant_id),
               data = results.dictator,
               family = beta_family)
summary(model.dictator)

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

# %%
m[m[, 4] <= 0.05, ]

# %%
odds.ratios.significant <- xtable(m[m[, 4] < 0.05, ], caption="Model estimates for significant coefficients only ($p<0.05$) on odds ratio scale")
print(odds.ratios.significant, type="latex")

# %%
xtable(coef(summary(model.pd))$cond, digits=3)

# %%
pd.predictions <- ggpredict(model.pd.1, c("Participant_group", "Model")) 

# %%
names(pd.predictions)[6] <- 'Model'
names(pd.predictions)

# %%
theoretical.data <- function(group, frequencies) {
    n = length(frequencies)
    result <- data.frame(
                                      predicted = frequencies, 
                                      std.error = rep(0, n),
                                      conf.low = frequencies,
                                      conf.high = frequencies,
                                      group = as.factor(rep('hypothesized', n)),
                                      facet = as.factor(rep(group, n))
                                     )
    if (n > 1) {
        result$x = as.factor(c('D', 'T4TD', 'T4TC', 'C'))
    }
    return(result)
}

theoretical.df <- function(group, fn = function(x) x) {
    generate.data <- function(frequencies) theoretical.data(group, fn(frequencies))
    case_when(
        group == 'Selfish'     ~ generate.data(c(0, 0, 0, 0)),
        group == 'Cooperative' ~ generate.data(c(1/6, 3/6, 1, 1)),
        group == 'Competitive' ~ generate.data(c(1/6, 1/6, 1/6, 1/6)),
        group == 'Altruistic'  ~ generate.data(c(1, 1, 1, 1)),
        group == 'Control'     ~ generate.data(c(NA, NA, NA, NA))
    )
}

# %%
groups <- levels(results.pd$Participant_group)
dfs <- lapply(groups, function(g) theoretical.df(g, fn = mean))
hypothesized <- reduce(dfs, rbind)
hypothesized

# %%
names(hypothesized)

# %%
names(hypothesized)[5] <- 'Model'
names(hypothesized)[6] <- 'x'

# %%
names(hypothesized)

# %%
names(pd.predictions)

# %%
c(2:6, 1)

# %%
pd.predictions.all <- rbind(pd.predictions, subset(hypothesized, select=c(6, 1:5)))

# %%
options(repr.plot.width = 20, repr.plot.height = 10)
pdf("figs/pd-predictions.pdf", width=8, height=8)
pd.predictions.plot <- ggplot(pd.predictions.all) + aes(x = x, y = predicted, group = Model) + 
        geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = .1, position = position_dodge(0.06)) +
        geom_line(aes(color=Model), size = 1) + scale_y_continuous(limits = c(0, 1)) +
        scale_color_brewer(palette = "Dark2", direction = -1) +
        labs(title = "Prisoners Dilemma", 
             x = "Participant group", y = "Probability of cooperation") +
        theme(legend.position = "bottom")
print(pd.predictions.plot)
dev.off()
pd.predictions.plot

# %%
options(repr.plot.width = 20, repr.plot.height = 10)
dictator.predictions.plot <- 
        ggplot(ggpredict(model.dictator, c("Participant_group", "Model"))) +
        aes(x = x, y = predicted, group = group) + 
        geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = .1, 
                          position = position_dodge(0.06)) +
        geom_line(aes(color=group), size = 1) + scale_y_continuous(limits = c(0, 1)) +
        scale_color_brewer(palette = "Dark2", direction = -1) +
        labs(title = "Dictator", 
             x = "Participant group", y = "Probability of cooperation") +
        theme(legend.position = "bottom")
dictator.predictions.plot

# %%
predictions.for <- function(model, participant.group) {
    ggpredict(model.pd.1, c("Partner_condition", 
                            sprintf("Model [%s]", model), 
                            sprintf("Participant_group [%s]", participant.group)))
}

predictions.for.m <- addMemoization(predictions.for)

# %%
interaction.plots <- function(participant.group, legend) {    
    predictions <- mclapply(levels(results.pd$Model), 
                            function(model) predictions.for.m(model, participant.group), 
                            mc.cores=12) 
    combined <- reduce(predictions, rbind)
    combined <- rbind(combined, theoretical.df(participant.group))
    names(combined)[6] <- 'Model'
    
    p <- ggplot(combined) +
        aes(x = x, y = predicted, group = Model) + 
        geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = .1, position = position_dodge(0.06)) +
        geom_line(aes(color=Model), size = 1) + scale_y_continuous(limits = c(0, 1)) +
        scale_color_brewer(palette = "Dark2", direction = -1) +       
        labs(title = sprintf("Group: %s", participant.group), 
             x = "Partner condition", y = "Probability of cooperation") +
         theme(legend.title = element_text(size=16), 
                legend.text = element_text(size=14)) 
    if (!legend) {
        p <- p + theme(legend.position = "none")
    } else {
        p <- p + theme(legend.title = element_text(size=16), 
                        legend.text = element_text(size=14))
    }
    return(p)
}

all.interaction.plots <- function() {
    groups <- levels(results.pd$Participant_group)
    all.plots <- mclapply(1:length(groups),            
                            function(g) interaction.plots(groups[g], legend = TRUE), 
                            mc.cores=12)
    result <- reduce(all.plots, function(x, y) x + y) + 
           guide_area() + 
           plot_layout(ncol=2) + 
           plot_layout(guides = 'collect') 
         
    return(result)
}

# %%
options(repr.plot.width = 20, repr.plot.height = 20)
combined.plots <- all.interaction.plots()
pdf("figs/interaction-plots-all.pdf", width=8, height=11)
print(combined.plots)
dev.off()
combined.plots

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
#simulationOutput.factorial<- simulateResiduals(fittedModel = model.factorial, plot = TRUE, integerResponse=TRUE)

# %%
#hist(residuals(simulationOutput.factorial))

# %%
#plotResiduals(simulationOutput.factorial, results.clean$Participant_group)

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
