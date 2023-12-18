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
                 Participant_group + Partner_condition + t + Model + Temperature +
                 Partner_condition:Model + Participant_group:Model +
                 (1|Participant_id),
               data = results.pd,
               family = betabinomial)
summary(model.pd)

# %%
texreg(model.pd, caption="Fitted model for Prisoners Dilemma", label="table:pd-estimates", file="pd-estimates.tex", single.row=TRUE, fontsize="small")

# %%
xtable(lme4::formatVC(summary(model.pd)$varcor$cond))

# %%
xtable(coef(summary(model.pd))$cond, digits=3)

# %%
options(repr.plot.height=10)
simulationOutput <- simulateResiduals(fittedModel = model.pd, plot = TRUE, integerResponse=TRUE)

# %%
hist(residuals(simulationOutput))

# %%
model.pd.factorial <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
                 Participant_group + Partner_condition + t + Model + Temperature +
                 Partner_condition:Model + Participant_group:Model + Participant_labels_reversed:Participant_label + Participant_labels_reversed:Model +
                 Participant_label + Participant_chain_of_thought + Participant_pronoun + Participant_defect_first + Participant_labels_reversed,
               data = results.clean,
               family = betabinomial)
summary(model.pd.factorial)

# %%
simulationOutput.factorial<- simulateResiduals(fittedModel = model.factorial, plot = TRUE, integerResponse=TRUE)

# %%
hist(residuals(simulationOutput.factorial))

# %%
plotResiduals(simulationOutput.factorial, results.clean$Participant_group)
