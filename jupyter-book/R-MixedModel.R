# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
libs <- c(
  "brms",
  "ggpubr",
  "reticulate",
  "lme4",
  "rstatix",
  "glmmTMB",
  "DHARMa",
  "modelsummary",
  "stargazer",
  "repr",
  "dplyr",
  "xtable",
  "texreg",
  "geepack",
  "ggplot2",
  "ggeffects",
  "MCMCglmm",
  "parallel",
  "coda",
  "effects",
  "ggeffects",
  "gridExtra",
  "purrr",
  "memoise",
  "R.cache",
  "patchwork",
  "ordinal",
  "cmdstanr",
  "checkmate"
)

lapply(libs, function(lib) library(lib, character.only = TRUE))

# %%
citations <- lapply(libs, function(lib) print(citation(lib), style = "Bibtex"))

# %%
options(repr.plot.width = 20, repr.plot.height = 10)

# %%
use_condaenv("llm-cooperation")
llm_coop <- import("llm_cooperation.main")

# %%
config <- llm_coop$Configuration(
  grid = dict(
    temperature = c(0.1, 0.6),
    model = c("gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"),
    max_tokens = list(500)
  ),
  experiment_names = c("dilemma", "dictator"),
  num_participant_samples = 30,
  num_replications = 3
)

# %%
results <- llm_coop$load_all(config)

# %%
ggboxplot(results,
  x = "Participant_group", y = "Cooperation frequency",
  color = "Partner Condition"
)

# %%
names(results)

# %%
names(results)[1] <- "Participant_condition"
names(results)[2] <- "Partner_condition"
names(results)[4] <- "Cooperation_frequency"

# %%
results_clean <- results %>%
  convert_as_factor(
    Experiment, Participant_group, Partner_condition,
    t, Model, Participant_id, Participant_prompt_index,
    Participant_chain_of_thought, Participant_label,
    Participant_case, Participant_pronoun,
    Participant_defect_first, Participant_labels_reversed
  ) %>%
  filter(!is.na(Cooperation_frequency))

# %%
levels(results_clean$Participant_pronoun)

# %%
levels(results_clean$Experiment)

# %%
levels(results_clean$Model)

# %%
levels(results_clean$Participant_group)

# %%
levels(results_clean$Partner_condition)

# %%
levels(results_clean$Partner_condition) <- c("T4TC", "T4TD", "C", "D")

# %%
levels(results_clean$Participant_label)

# %%
levels(results_clean$Participant_case)

# %%
results_clean$Participant_group <-
  relevel(results_clean$Participant_group, ref = "Control")

results_clean$Partner_condition <-
  relevel(results_clean$Partner_condition, ref = "D")

results_clean$Model <-
  relevel(results_clean$Model, ref = "gpt-3.5-turbo-0613")

results_clean$Participant_pronoun <-
  relevel(results_clean$Participant_pronoun, ref = "they")

results_clean$Participant_case <-
  relevel(results_clean$Participant_case, ref = "standard")

results_clean$Particpant_label <-
  relevel(results_clean$Participant_label, ref = "colors")

# %%
levels(results_clean$Partner_condition)

# %%
results_clean$Partner_condition <-
  factor(results_clean$Partner_condition,
    levels = c("D", "T4TD", "T4TC", "C")
  )

# %%
levels(results_clean$Partner_condition)

# %%
results_clean$Participant_group <-
  factor(results_clean$Participant_group,
    levels = c("Control", "Selfish", "Competitive", "Cooperative", "Altruistic")
  )

# %%
levels(results_clean$Participant_group)

# %%
names(results_clean)

# %%
results_clean <- results_clean %>%
  select(-Choices, -Transcript, -Participant_condition)

# %%
names(results_clean)

# %%
hist(results_clean$Cooperation_frequency)

# %%
hist(
  results_clean[results_clean$Experiment == "dictator", ]$Cooperation_frequency
)

# %%
hist(
  results_clean[results_clean$Experiment == "dilemma", ]$Cooperation_frequency
)

# %%
stargazer(results_clean)

# %%
summary.stats <- results_clean %>%
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
results_grouped <- results_clean %>%
  group_by(
    Experiment,
    Participant_group,
    Partner_condition,
    t,
    Model,
    Temperature
  )
head(results_grouped)

# %%
options(repr.plot.height = 24)

ggplot(results_clean, aes(x = Participant_group, y = Cooperation_frequency)) +
  geom_boxplot() +
  facet_wrap(
    . ~ Model + Temperature + Participant_labels_reversed + Partner_condition,
    ncol = 6
  )

# %%
results_pd <- results_clean[results_clean$Experiment == "dilemma", ]

head(results_pd)
# %%
results_dictator <- results_clean[results_clean$Experiment == "dictator", ]
head(results_dictator)

# %%
results_pd$Num_cooperates <- round(results_pd$Cooperation_frequency * 6)

# %%
num_defects <- 6 - results_pd$Num_cooperates
summary(num_defects)

# %%
verbatim <- function(x) {
  txt <- trimws(capture.output(x))
  txt <- c("\\begin{verbatim}", txt, "\\end{verbatim}")
  cat(txt, sep = "\n")
}

# %%
results_dictator$Num_cooperates <- results_dictator$Cooperation_frequency * 4
results_dictator$Num_defects <- 4 - results_dictator$Cooperation_frequency * 4
summary(results_dictator$Num_cooperates)

# %%
levels(results_dictator$Model)

# %%
dictator_counts <- results_dictator %>% count(Num_cooperates)

# %%
pdf("figs/dictator-num-cooperates-hist.pdf")
ggplot(dictator_counts, aes(x = Num_cooperates, y = n)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Histogram of Dictator Game responses",
    x = "Donation amount",
    y = "Frequency"
  )
dev.off()

# %%
results_dictator$Response <- as.factor(results_dictator$Num_cooperates)

# %%

model_dictator <- clmm(
  Response ~
    Participant_group + Participant_group:Model + t + Model + Temperature +
    (1 | Participant_id),
  link = "logit", threshold = "equidistant",
  data = results_dictator
)
summary(model_dictator)

# f%%

model_dictator_hess <- clmm(
  Response ~
    Participant_group + Participant_group:Model + t + Model + Temperature +
    (1 | Participant_id),
  link = "logit", threshold = "equidistant",
  data = results_dictator,
  Hess = TRUE,
  nAGQ = 10
)
summary(model_dictator_hess)

# %% [markdown]

# Remove t parameter from the model since it had insigicant estimate and is not expected to have any causal effect.

# %%
model_dictator_1 <- clmm(
  Response ~
    Participant_group + Participant_group:Model + Model + Temperature +
    (1 | Participant_id),
  link = "logit", threshold = "equidistant",
  data = results_dictator
)

# %%
summary(model_dictator_1)

# %%
print(odds_ratios_significant(model_dictator_1), type = "latex")

# %%
model_pd_poisson <- glmmTMB(
  Num_cooperates ~
    Participant_group + Participant_group:Partner_condition + t +
    Model + Temperature +
    Participant_group:Partner_condition:Model +
    (1 | Participant_id),
  data = results_pd,
  family = poisson
)
summary(model_pd_poisson)

# %%
model_pd <- glmmTMB(
  cbind(Num_cooperates, 6 - Num_cooperates) ~
    Participant_group + Participant_group:Partner_condition +
    t + Model + Temperature +
    (1 | Participant_id),
  data = results_pd,
  family = betabinomial
)

# %%
summary(model_pd)

# %%
model_pd_1 <- glmmTMB(
  cbind(Num_cooperates, 6 - Num_cooperates) ~
    Participant_group + Participant_group:Partner_condition +
    t + Model + Temperature +
    Participant_group:Partner_condition:Model +
    (1 | Participant_id),
  data = results_pd,
  family = betabinomial
)
summary(model_pd_1)

# %%
model_pd_2 <- glmmTMB(
  cbind(Num_cooperates, 6 - Num_cooperates) ~
    Participant_group + Participant_group:Partner_condition +
    Model + Temperature +
    Participant_group:Partner_condition:Model +
    (1 | Participant_id),
  data = results_pd,
  family = betabinomial
)
summary(model_pd_2)


# %%
texreg(

  model_pd_1,
  caption = "Fitted model for Prisoners Dilemma",
  label = "table:pd-estimates",
  file = "pd-estimates.tex",
  single.row = TRUE,
  fontsize = "small"
)

# %%
texreg(
  model_dictator,
  caption = "Fitted model for Dictator",
  label = "table:dictator-estimates",
  file = "dictator-estimates.tex",
  single.row = TRUE,
  fontsize = "small"
)

# %%
pdf("figs/ranef_hist_pd.pdf")
hist(ranef(model_pd_1)$cond$Participant_id[, 1])
dev.off()

# %%
hist(ranef(model_dictator_1)$Participant_id[, 1])
pdf("figs/ranef_hist_dictator.pdf")
dev.off()
# %%
xtable(lme4::formatVC(summary(model_pd)$varcor$cond))

# %%
as_odds <- function(model) {
  m <- coef(summary(model))
  if (testClass(m, "list")) {
    m <- m$cond
  }
  m[, 1] <- exp(m[, 1])
  m[, 2] <- exp(m[, 2])
  colnames(m)[1] <- "Estimate (Odds ratio)"
  return(m)
}

# %%
as_odds(model_pd_1)

# %%
odds_ratios_significant <- function(model) {
  m <- as_odds(model)
  xtable(
    m[m[, 4] < 0.05, ],
    caption =
      "Model estimates for significant coefficients only ($p<0.05$) on odds ratio scale"
  )
}

# %%

odds_ratios_significant(model_pd_1)

# %%
print(odds_ratios_significant(model_pd_1), type = "latex")

# %%
xtable(coef(summary(model_pd))$cond, digits = 3)

# %%
predictions_by_group <- function(model) {
  ggpredict(model, c("Participant_group", "Model")) %>%
    rename_at("group", ~"Model")
}

# %%
pd_predictions <- predictions_by_group(model_pd_2)

# %%
theoretical_data <- function(group, frequencies) {
  n <- length(frequencies)
  result <- data.frame(
    predicted = frequencies,
    std.error = rep(0, n),
    conf.low = frequencies,
    conf.high = frequencies,
    group = as.factor(rep("hypothesized", n)),
    facet = as.factor(rep(group, n))
  )
  if (n > 1) {
    result$x <- as.factor(c("D", "T4TD", "T4TC", "C"))
  }
  return(result)
}

theoretical_dilemma <- function(group) {
  case_when(
    group == "Selfish" ~ c(0, 0, 0, 0),
    group == "Cooperative" ~ c(1 / 6, 3 / 6, 1, 1),
    group == "Competitive" ~ c(1 / 6, 1 / 6, 1 / 6, 1 / 6),
    group == "Altruistic" ~ c(1, 1, 1, 1),
    group == "Control" ~ c(NA, NA, NA, NA)
  )
}

theoretical_dictator <- function(group) {
  case_when(
    group == "Selfish" ~ 0,
    group == "Cooperative" ~ 0.5,
    group == "Competitive" ~ 0,
    group == "Altruistic" ~ 1.0,
    group == "Control" ~ NA
  )
}

theoretical_df_for_group <- function(
    group, fn = function(x) x, experiment = theoretical_dilemma) {
  d <- experiment(group)
  theoretical_data(group, fn(d))
}

theoretical_df <- function(experiment) {
  groups <- levels(results_pd$Participant_group)
  dfs <- lapply(
    groups,
    function(g) theoretical_df_for_group(g, fn = mean, experiment = experiment)
  )
  hypothesized <- reduce(dfs, rbind)
  names(hypothesized)[5] <- "Model"
  names(hypothesized)[6] <- "x"
  return(hypothesized)
}

# %%
hypothesized_dilemma <- theoretical_df(experiment = theoretical_dilemma)
hypothesized_dilemma

# %%
hypothesized_dictator <- theoretical_df(experiment = theoretical_dictator)
hypothesized_dictator

# %%
pd_predictions_poisson <- predictions_by_group(model_pd_poisson)
pd_predictions_poisson$predicted <- pd_predictions_poisson$predicted / 6
pd_predictions_poisson$conf.low <- pd_predictions_poisson$conf.low / 6
pd_predictions_poisson$conf.high <- pd_predictions_poisson$conf.high / 6
pd_predictions_poisson

# %%
actual_and_theoretical <- function(predictions, h = hypothesized_dilemma) {
  rbind(predictions, subset(h, select = c(6, 1:5)))
}

# %%
actual_and_theoretical(pd_predictions)

# %%
pd_predictions
# %%
predictions_plot <- function(predictions, title) {
  ggplot(predictions) +
    aes(x = x, y = predicted, group = Model) +
    geom_errorbar(
      aes(ymin = conf.low, ymax = conf.high),
      width = .1, position = position_dodge(0.06)
    ) +
    geom_ribbon(
      aes(x = x, y = predicted, ymax = conf.high, ymin = conf.low, fill = Model),
      alpha = 0.2
    ) +
    geom_line(aes(color = Model), size = 1) + # scale_y_continuous(limits = c(0, 1)) +
    scale_color_brewer(palette = "Dark2", direction = -1) +
    scale_fill_brewer(palette = "Dark2", direction = -1, guide = "none") +
    labs(
      title = title,
      x = "Participant group", y = "Level of cooperation"
    ) +
    theme(legend.position = "bottom")
}

# %%
options(repr.plot.width = 20, repr.plot.height = 10)
pdf("figs/pd-predictions.pdf", width = 8, height = 8)
pd_predictions_plot <-
  predictions_plot(
    actual_and_theoretical(pd_predictions),
    "Prisoners Dilemma"
  )
print(pd_predictions_plot)
dev.off()

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
predictions_dictator <- predictions_by_group(model_dictator_1)

# %%
weighted_dictator <- predictions_dictator %>%
  group_by(x, Model) %>%
  summarise(
    predicted = weighted.mean(predicted, as.numeric(response.level) - 1),
    conf.high = weighted.mean(conf.high, as.numeric(response.level) - 1),
    conf.low = weighted.mean(conf.low, as.numeric(response.level) - 1),
  )

# %%
xtable(weighted_dictator, digits = 3)

# %%
predictions_dictator <- actual_and_theoretical(weighted_dictator, h = hypothesized_dictator)
predictions_dictator

# %%
dictator_predictions_plot <-
  predictions_plot(predictions_dictator, "Dictator Game")
dictator_predictions_plot

# %%
options(repr.plot.width = 20, repr.plot.height = 10)
cooperation_by_group_plots <-
  dictator_predictions_plot + pd_predictions_plot +
  plot_layout(ncol = 2) + plot_layout(guides = "keep")
pdf("figs/cooperation-by-group.pdf", width = 16, height = 8)
print(cooperation_by_group_plots)
dev.off()
cooperation_by_group_plots

# %%
predictions_for <- function(gpt_model, participant_group, model = model_pd_1) {
  ggpredict(model, c(
    "Partner_condition",
    sprintf("Model [%s]", gpt_model),
    sprintf("Participant_group [%s]", participant_group)
  ))
}

predictions_for_m <- addMemoization(predictions_for)

# %%
interaction_plots <- function(participant_group, legend) {
  predictions <- mclapply(levels(results_pd$Model),
    function(model) predictions_for_m(model, participant_group),
    mc.cores = 12
  )
  combined <- reduce(predictions, rbind)
  combined <- rbind(
    combined,
    theoretical_df_for_group(participant_group, experiment = theoretical_dilemma)
  )
  names(combined)[6] <- "Model"

  p <- ggplot(combined) +
    aes(x = x, y = predicted, group = Model) +
    geom_ribbon(
      aes(x = x, y = predicted, ymax = conf.high, ymin = conf.low, fill = Model),
      alpha = 0.2
    ) +
    geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = .1, position = position_dodge(0.16)) +
    geom_line(aes(color = Model), size = 1) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_brewer(palette = "Dark2", direction = -1) +
    scale_fill_brewer(palette = "Dark2", direction = -1, guide = "none") +
    labs(
      title = sprintf("Group: %s", participant_group),
      x = "Partner condition", y = "Probability of cooperation"
    ) +
    theme(
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 14)
    )
  if (!legend) {
    p <- p + theme(legend.position = "none")
  } else {
    p <- p + theme(
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 14)
    )
  }
  return(p)
}

all_interaction_plots <- function() {
  groups <- levels(results_pd$Participant_group)
  all.plots <- mclapply(1:length(groups),
    function(g) interaction_plots(groups[g], legend = TRUE),
    mc.cores = 12
  )
  result <- reduce(all.plots, function(x, y) x + y) +
    guide_area() +
    plot_layout(ncol = 2) +
    plot_layout(guides = "collect")

  return(result)
}

# %%
options(repr.plot.width = 20, repr.plot.height = 20)
combined.plots <- all_interaction_plots()
pdf("figs/interaction-plots-all.pdf", width = 8, height = 11)
print(combined.plots)
dev.off()
combined.plots

# %%
options(repr.plot.height = 12)
pdf("figs/residuals-plots.pdf")
simulationOutput <-
  simulateResiduals(fittedModel = model_pd_1, plot = TRUE, integerResponse = TRUE)
dev.off()

# %%
pdf("figs/residuals-hist.pdf")
hist(residuals(simulationOutput))
dev.off()

# %%
# Fit a GEE model with an interaction term
gee_model <- geeglm(
  cbind(Num_cooperates, 6 - Num_cooperates) ~ Participant_group + Partner_condition + t + Model + Temperature +
    Partner_condition:Model + Participant_group:Model,
  id = Participant_id,
  data = results_pd,
  family = binomial,
  corstr = "exchangeable"
)

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
# model.pd.factorial <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
#                 Participant_group + Partner_condition + t + Model + Temperature +
#                 Partner_condition:Model + Participant_group:Model + Participant_labels_reversed:Participant_label + Participant_labels_reversed:Model +
#                 Participant_label + Participant_chain_of_thought + Participant_pronoun + Participant_defect_first + Participant_labels_reversed,
#               data = results.clean,
#               family = betabinomial)
# summary(model.pd.factorial)

# %%
# simulationOutput.factorial<- simulateResiduals(fittedModel = model.factorial, plot = TRUE, integerResponse=TRUE)

# %%
# hist(residuals(simulationOutput.factorial))

# %%
# plotResiduals(simulationOutput.factorial, results.clean$Participant_group)

# %% [markdown]
# ## Bayesian MCMC GLMM

# %%
# STEP 1: DEFINE the model
bb_model <- "
  data {
    int<lower = 0, upper = 10> Y;
  }
  parameters {
    real<lower = 0, upper = 1> pi;
  }
  model {
    Y ~ binomial(10, pi);
    pi ~ beta(2, 2);
  }
"

# %%
library(rstan)
# STEP 2: SIMULATE the posterior
bb_sim <- stan(
  model_code = bb_model, data = list(Y = 9),
  chains = 4, iter = 5000 * 2, seed = 84735
)

# %%
library(bayesplot)
# Histogram of the Markov chain values
mcmc_hist(bb_sim, pars = "pi") +
  yaxis_text(TRUE) +
  ylab("count")


# %%
pr <- prior(normal(0, 1), class = "b")
brm_model_dictator <- brm(
  Num_cooperates | trials(4) ~
    Participant_group + t + Model + Model:Participant_group,
  ~ (1 | Participant_id),
  data = results_dictator,
  prior = NULL,
  family = mixture(beta_binomial, poisson),
  chains = 8, # The default number of chains in brms
  cores = 8,
  iter = 4000, # The default number of iterations per chain
  # warmup = 1000,       # The default number of warmup iterations
  # seed = 101,         # Set a seed for reproducibility
  # backend = "cmdstanr"
)

# %%
summary(brm_model_dictator)

# %%
plot(brm_model_dictator)

# %%
pr <- prior(normal(0, 1), class = "b")
brm_model <- brm(
  Num_cooperates | trials(6) ~
    Participant_group + Partner_condition + t + Model +
    Model:Partner_condition:Participant_group,
  ~ (1 | Participant_id),
  data = results_pd,
  prior = NULL,
  # prior = pr,
  family = beta_binomial, # Using beta binomial with logit link
  chains = 8, # The default number of chains in brms
  cores = 8,
  iter = 4000, # The default number of iterations per chain
  # warmup = 1000,       # The default number of warmup iterations
  # seed = 101,         # Set a seed for reproducibility
  # backend = "cmdstanr"
)

# %%
summary(brm_model)

# %%

# %%
plot(brm_model)

# %%
brms::pp_check(brm_model_dictator, type = "ecdf_overlay", ndraws = 100)

# %%
brms::conditional_effects(brm_model)

# %%
brms::pp_check(brm_model, type = "ecdf_overlay", ndraws = 100)

# %%
posterior_summary(brm_model)

# %%
# Set up a list for priors
prior <- list(
  R = list(V = 1, nu = 0.002), # Prior for the residual covariance matrix
  G = list(G1 = list(V = 1, nu = 0.002)) # Prior for the random effects
)

estimate_glmm <- function() {
  # Fit the model with the specified priors
  MCMCglmm(
    fixed =
      Num_cooperates ~ Participant_group + Partner_condition + t +
        Model + Model:Partner_condition:Participant_group,
    random = ~ us(1):Participant_id,
    prior = prior,
    data = results_pd,
    verbose = FALSE
  )
}

model_pd_mcmc <- estimate_glmm()

# %%
plot(model_pd_mcmc$Sol)

# %%
plot(model_pd_mcmc$VCV)

# %%
m6 <- mclapply(1:12, function(i) {
  estimate_glmm()
}, mc.cores = 12)
m6 <- lapply(m6, function(m) m$Sol)
m6 <- do.call(mcmc.list, m6)

# %%
par(mfrow = c(4, 2), mar = c(2, 2, 1, 2))
gelman.plot(m6, auto.layout = FALSE)

# %%
gelman.diag(m6)

# %%
par(mfrow = c(8, 2), mar = c(2, 1, 1, 1))
plot(m6, ask = FALSE, auto.layout = FALSE)

# %%
summary(m6)

# %%
# plot_estimates <- function(x) {
#   if (class(x) != "summary.mcmc") {
#     x <- summary(x)
#   }
#   n <- dim(x$statistics)[1]
#   par(mar = c(2, 7, 4, 1))
#   plot(x$statistics[, 1], n:1,
#     yaxt = "n", ylab = "",
#     xlim = range(x$quantiles) * 1.2,
#     pch = 19,
#     main = "Posterior means and 95% credible intervals"
#   )
#   grid() axis(2, at = n:1, rownames(x$statistics), las = 2)
#   arrows(x$quantiles[, 1], n:1, x$quantiles[, 5], n:1, code = 0)
#   abline(v = 0, lty = 2)
# }

# options(repr.plot.height = 24, repr.plot.width = 20)
# plot_estimates(m6)
