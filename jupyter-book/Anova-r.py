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
library(rstatix)

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
results.clean <- results %>% convert_as_factor(Participant_group, Partner_condition, t, Model, Participant_id, Temperature) %>% filter(!is.na(Cooperation_frequency))

# %%
res.aov <- anova_test(data=results.clean, dv=Cooperation_frequency, wid=Participant_id, between=Participant_group, within=c(Temperature, t, Partner_condition, Model))
get_anova_table(res.aov)

# %%
