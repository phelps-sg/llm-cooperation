# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Results

# %% tags=["hide-input"]
import pandas as pd
import plotly.express as px
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from llm_cooperation.experiments.dictator import *
from llm_cooperation.experiments.dilemma import *
from llm_cooperation.experiments.principalagent import *
from llm_cooperation.main import load_all, get_config, Configuration
from llm_cooperation import Choice
from llm_cooperation.notebook import graph, save_table
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt

# %%
logging.basicConfig(level=logging.INFO)

# %%
config = Configuration(
    grid={
        "temperature": [0.1, 0.6],
        "model": ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"],
        "max_tokens": [500],
    },
    experiment_names=["dilemma"],
    num_participant_samples=30,
    num_replications=3,
)
config

# %%
results = load_all(config)
results

# %%
results.to_csv("results/pd-all.csv")

# %% [markdown]
# ## Anova

# %%
results = results[~results["Cooperation frequency"].isnull()]

# %%
logging.basicConfig(level=logging.INFO)
plt.figure(figsize=(10, 6))
fig = interaction_plot(
    x=results["Participant_group"],
    trace=results["Partner Condition"],
    response=results["Cooperation frequency"],
)


# %%
plt.figure(figsize=(10, 6))
fig = interaction_plot(
    x=results["Participant_group"],
    trace=results["Model"],
    response=results["Cooperation frequency"],
)

# %%
plt.figure(figsize=(10, 6))
fig = interaction_plot(
    x=results["Participant_group"],
    trace=results["Temperature"],
    response=results["Cooperation frequency"],
)

# %%
N = len(results)

# %% [markdown]
# The data consists of a total of

# %% tags=["hide-input"]
print(f"N = {N}")

# %% [markdown]
# cases.  Each case corresponds to a single play of the iterated PD over six rounds.  For each case we record the following fields:

# %% tags=["hide-input"]
results.iloc[0]

# %%
results.groupby("Participant_id").describe()

# %% [markdown]
# ### Table 1: Cooperation frequency by group

# %% tags=["hide-input"]
table1 = (
    results.groupby("Participant_group")["Cooperation frequency"].describe().round(2)
)
save_table(table1, "table1", "Cooperation frequency by group")

# %% [markdown]
# ### Table 2: Cooperation frequency by group/condition

# %% tags=["hide-input"]
table2 = (
    results.groupby(["Participant_group", "Partner Condition"])["Cooperation frequency"]
    .describe()
    .round(2)
)
save_table(table2, "table2", "Cooperation frequency by group/condition")

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Model",
        notched=True,
    ),
    "pd-boxplot-group-model",
)

# %%
graph(
    lambda: px.box(
        results, x="Participant_group", y="Cooperation frequency", notched=True
    ),
    "pd-boxplot-group",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Temperature",
        notched=True,
    ),
    "pd-boxplot-group-temperature",
)

# %%
graph(
    lambda: px.box(results, x="Model", y="Cooperation frequency", notched=True),
    "pd-boxplot-model",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Participant_pronoun",
        notched=True,
    ),
    "pd-boxplot-group-pronoun",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Participant_case",
        notched=True,
    ),
    "pd-boxplot-group-case",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Participant_label",
        notched=True,
    ),
    "pd-boxplot-group-label",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Participant_defect_first",
        notched=True,
    ),
    "pd-boxplot-group-defect-first",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Model",
        notched=True,
    ),
    "pd-boxplot-partner-condition-model",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_labels_reversed",
        notched=True,
    ),
    "pd-boxplot-partner-condition-labels-reversed",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Model",
        y="Cooperation frequency",
        color="Participant_labels_reversed",
        notched=True,
    ),
    "pd-boxplot-model-labels-reversed",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_label",
        y="Cooperation frequency",
        color="Participant_labels_reversed",
        notched=True,
    ),
    "pd-boxplot-label-labels-reversed",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-partner-condition-cot",
)

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_chain_of_thought",
        color="Temperature",
        y="Cooperation frequency",
        notched=True,
    ),
    "pd-boxplot-cot-temperature",
)

# %% tags=["hide-input"]
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-group-cot",
)

# %% tags=["hide-input"]
graph(
    lambda: px.box(
        results,
        x="Participant_group",
        y="Cooperation frequency",
        color="Temperature",
        notched=True,
    ),
    "pd-boxplot-group-temperature",
)


# %%
def fig_group_partner_condition(model=None):
    results_sel = results[results.Model == model] if model is not None else results
    model_str = "all" if model is None else model
    graph(
        lambda: px.box(
            results_sel,
            x="Participant_group",
            y="Cooperation frequency",
            color="Partner Condition",
            notched=True,
        ),
        f"pd-boxplot-group-partner-condition-{model_str}",
    )


# %%
fig_group_partner_condition()

# %%
fig_group_partner_condition(model="gpt-3.5-turbo-0301")

# %%
fig_group_partner_condition(model="gpt-3.5-turbo-0613")

# %%
fig_group_partner_condition(model="gpt-3.5-turbo-1106")

# %%
graph(
    lambda: px.box(
        results,
        x="Participant_chain_of_thought",
        y="Cooperation frequency",
        color="Partner Condition",
        notched=True,
    ),
    "pd-boxplot-cot-partner-condition",
)

# %%
results_sel = results[results["Participant_group"] == "Cooperative"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Model",
        notched=True,
    ),
    "pd-boxplot-partner-condition-model",
)

# %%
results_sel = results[results["Participant_group"] == "Cooperative"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-cooperative-group-partner-condition-cot",
)

# %%
results_sel = results[results["Participant_group"] == "Selfish"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-selfish-group-partner-condition-cot",
)

# %%
results_sel = results[results["Participant_group"] == "Altruistic"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-altruistic-group-partner-condition-cot",
)

# %%
results_sel = results[results["Participant_group"] == "Altruistic"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Model",
        notched=True,
    ),
    "pd-boxplot-altruistic-group-partner-condition-model",
)

# %%
results_sel = results[
    (results["Participant_group"] == "Altruistic")
    & (results["Model"] == "gpt-3.5-turbo-1106")
]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-altruistic-group-1106-partner-condition-cot",
)

# %%
results_sel = results[results["Participant_group"] == "Competitive"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-competitive-group-partner-condition-cot",
)

# %%
results_sel = results[results["Participant_group"] == "Control"]
graph(
    lambda: px.box(
        results_sel,
        x="Partner Condition",
        y="Cooperation frequency",
        color="Participant_chain_of_thought",
        notched=True,
    ),
    "pd-boxplot-control-group-partner-condition-cot",
)
