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
# # Principal-agent

# %% [markdown]
# ## Results

# %% tags=["hide-input"]
import pandas as pd
import plotly.express as px
from llm_cooperation.experiments.dictator import *
from llm_cooperation.experiments.dilemma import *
from llm_cooperation.experiments.principalagent import *
from llm_cooperation.main import load_all, get_config, Configuration
from llm_cooperation import Choice
from llm_cooperation.notebook import graph

# %%
config = Configuration(
    grid={
        "temperature": [0.1, 0.6],
        "model": [
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4",
        ],
        "max_tokens": [500],
    },
    experiment_names=["principal-agent"],
    num_participant_samples=30,
    num_replications=3,
)
config

# %%
results = load_all(config)
results

# %%
results["info_condition"] = [None] * len(results)
results.loc[
    results.Participant_shared_with_user & results.Participant_shared_with_principal,
    "info_condition",
] = "both"
results.loc[
    ~(results.Participant_shared_with_user & results.Participant_shared_with_principal),
    "info_condition",
] = "neither"
results.loc[
    results.Participant_shared_with_user & ~results.Participant_shared_with_principal,
    "info_condition",
] = "user-only"
results.loc[
    ~results.Participant_shared_with_user & results.Participant_shared_with_principal,
    "info_condition",
] = "principal-only"

# %%
results["info_condition"]

# %%
results["ChoiceValue"] = results.Choice.map(
    lambda x: x.value if x is not None else np.nan
)

# %%
selection = results.Participant_principal == "OpenAI"
graph(
    lambda: px.box(
        results[selection],
        x="info_condition",
        y="ChoiceValue",
        color="Model",
        notched=True,
    ),
    "pa-figure2",
)

# %%
selection = results.Participant_principal == "Shell Oil"
graph(
    lambda: px.box(
        results[selection],
        x="info_condition",
        y="ChoiceValue",
        color="Model",
        notched=True,
    ),
    "pa-figure3",
)

# %%
selection = (results.Participant_principal == "Shell Oil") & (
    results.Model == "gpt-3.5-turbo-0301"
)
graph(
    lambda: px.box(
        results[selection],
        x="info_condition",
        y="ChoiceValue",
        color="Participant_case",
    ),
    "pa-figure4",
)

# %%
selection = (results.Participant_principal == "Shell Oil") & (
    results.Model == "gpt-3.5-turbo-0301"
)
graph(
    lambda: px.box(
        results[selection],
        x="info_condition",
        y="ChoiceValue",
        color="Participant_labels_reversed",
    ),
    "pa-figure5",
)
