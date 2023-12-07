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
results.clean <- results %>% convert_as_factor(Participant_group, Partner_condition, t, Model, Participant_id) %>% filter(!is.na(Cooperation_frequency))

# %%
results.clean <- results.clean[,c("Participant_group", "Partner_condition", "t", "Model", "Participant_id", "Temperature", "Cooperation_frequency")]
results.clean

# %%
levels(results.clean$Model)

# %%
levels(results.clean$Participant_group)

# %%
levels(results.clean$Partner_condition)

# %%
results.clean$Participant_group <- relevel(results.clean$Participant_group, ref='Control')
results.clean$Partner_condition <- relevel(results.clean$Partner_condition, ref='tit for tat D')
results.clean$Model <- relevel(results.clean$Model, ref='gpt-3.5-turbo-0613')

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
model <- glmmTMB(Cooperation_frequency ~ Participant_group + Partner_condition + t + Model + Temperature +
               (1 | Participant_id), 
               data = results.clean, 
               family = beta_family(link = "logit"))

summary(model)

# %%
