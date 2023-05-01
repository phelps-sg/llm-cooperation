# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Results

# %% tags=["hide-input"]
import pandas as pd

# %% tags=["hide-input"]
from dilemma import *

results = pd.read_pickle("../results-3.pickle")

# %%
results

# %% [markdown]
# ### Table 1: Cooperation frequency by participant

# %% tags=["hide-input"]
results.groupby("Participant").describe().round(2)

# %%
results.groupby(["Participant", "Condition"]).describe().round(2)

# %% [markdown]
# ### Table 2: Cooperation frequency by group

# %% tags=["hide-input"]
results.groupby("Group")["Cooperation frequency"].describe().round(2)

# %% [markdown]
# ### Table 4: Cooperation frequency by group/condition

# %% tags=["hide-input"]
results.groupby(["Group", "Condition"])["Cooperation frequency"].describe().round(2)

# %% tags=["hide-input"]
import plotly.express as px
import plotly
from IPython.core.display import HTML
from typing import Callable


def graph(fn: Callable, name: str):
    fig = fn()
    fname = f"{name}.html"
    fig.show()
    # plotly.offline.plot(fig, filename=fname, auto_open=False)
    # display(HTML(filename=fname))


def boxplot(group: str, name: str):
    graph(
        lambda: px.box(
            results[results.Group == group], x="Condition", y="Cooperation frequency"
        ),
        name,
    )


# %% [markdown]
# ### Figure 1: Cooperation frequency by group

# %% tags=["hide-input"]
results[["Group", "Condition", "Cooperation frequency"]].groupby("Group").boxplot(
    subplots=False, figsize=(20, 8)
)

# %% [markdown]
# ### Figure 2: Cooperation frequency by condition- control group

# %% tags=["hide-input"]

# %%
results[results.Group == "Group.Altruistic"][
    ["Group", "Condition", "Cooperation frequency"]
].groupby("Condition").boxplot(subplots=False, figsize=(20, 6))

# %%
cases = results[
    (results.Group == "Group.Altruistic")
    & (results.Condition == "unconditional cooperate")
]
cases

# %%
for line in cases.iloc[0].Transcript:
    print(line)

# %%
[c.user for c in cases.iloc[0].Choices]

# %%
[c.ai for c in cases.iloc[0].Choices]

# %%
for line in cases.iloc[1].Transcript:
    print(line)

# %%
cases = results[
    (results.Group == "Group.Altruistic")
    & (results.Condition == "unconditional defect")
]
cases

# %%
cases = results[
    (results.Group == "Group.Mixed") & (results.Condition == "tit for tat C")
]
cases

# %%
for line in cases.iloc[1].Transcript:
    print(line)

# %% [markdown]
# ### Figure 3: Cooperation frequency by condition- altruistic group

# %% tags=["hide-input"]
boxplot("Group.Altruistic", "figure3")

# %% [markdown]
# ### Figure 4: Cooperation frequency by condition- selfish group

# %% tags=["hide-input"]
boxplot("Group.Selfish", "figure4")

# %% [markdown]
# ### Figure 5: Cooperation frequency by condition- mixed group

# %% tags=["hide-input"]
boxplot("Group.Mixed", "figure5")
