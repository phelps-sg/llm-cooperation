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

# %%
llm.coop <- import("llm_cooperation.main")

# %%
config=llm.coop$Configuration(
    grid=dict(
        temperature=c(0.1, 0.6),
        model=c("gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"),
        max_tokens=list(500)
    ),
    experiment_names=list("dilemma"),
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
levels(results.clean$Experiment)

# %%
levels(results.clean$Model)

# %%
levels(results.clean$Participant_group)

# %%
levels(results.clean$Partner_condition)

# %%
levels(results.clean$Participant_pronoun)

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
results.clean$Particpant_labe <- relevel(results.clean$Participant_label, ref='colors')

# %%
hist(results.clean$Cooperation_frequency)

# %%
results %>% group_by(Participant_group, Partner_condition, t, Model, Temperature) %>% shapiro_test(Cooperation_frequency)

# %%
res.aov <- anova_test(data=results.clean, dv=Cooperation_frequency, wid=Participant_id, between=Participant_group, within=c(Temperature, t, Partner_condition, Model))
get_anova_table(res.aov)

# %%
# Adjust 0s and 1s
epsilon <- .Machine$double.eps ^ 0.5
results.clean$Cooperation_frequency[results.clean$Cooperation_frequency == 0] <- epsilon
results.clean$Cooperation_frequency[results.clean$Cooperation_frequency == 1] <- 1 - epsilon

# %%
results.clean$Num_cooperates = round(results.clean$Cooperation_frequency * 6)

# %%
6 - results.clean$Num_cooperates

# %%
model <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
                 Participant_group + Partner_condition + t + Model + Temperature +
                 Partner_condition:Model + Participant_group:Model +
                 (1|Participant_id),
               data = results.clean,
               family = betabinomial)
summary(model)

# %%
model <- glmmTMB(cbind(Num_cooperates, 6 - Num_cooperates)  ~
                 Participant_group + Partner_condition + t + Model + Temperature +
                 Partner_condition:Model + Participant_group:Model + Participant_labels_reversed:Participant_label + Participant_labels_reversed:Model +
                 Participant_label + Participant_chain_of_thought + Participant_pronoun + Participant_defect_first + Participant_labels_reversed,
               data = results.clean,
               family = betabinomial)
summary(model)

# %%
simulationOutput <- simulateResiduals(fittedModel = model, plot = TRUE, integerResponse=TRUE)

# %%
options(repr.plot.width = 20, repr.plot.height = 10)

# %%
testQuantiles(simulationOutput)

# %%
hist(residuals(simulationOutput))

# %%
plotResiduals(simulationOutput, results.clean$Participant_group)
