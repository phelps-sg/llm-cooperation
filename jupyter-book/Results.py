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
from llm_cooperation.experiments.dictator import *
from llm_cooperation import Choice

# %%
results = pd.read_pickle("../results/dictator-gpt-4.pickle")
results

# %%
colors = results.Choice.map(lambda x: x.payoff_allo if x is not None else None)

# %%
black_example = results[results.Choice == BLACK].iloc[0].Choice

# %%
black_examp

# %%
colors

# %%
colors

# %%
colors = pd.DataFrame(colors)
colors[colors["Choice"] == "project black"]

# %%
table1 = results.groupby("Group")["Cooperation frequency"].describe().round(2)
table1

# %%
table2 = (
    results.groupby(["Group", "Participant"])["Cooperation frequency"]
    .describe()
    .round(2)
)
table2

# %%
first_row = results.iloc[1]
first_row

# %%
first_row.Group

# %%
first_row.Participant

# %%
first_row.Transcript

# %%
altruist = results[results.Group == "Group.Altruistic"].iloc[0]
altruist

# %%
altruist.Transcript

# %%
import numpy as np

donation = [
    choice.payoff_allo
    for choice in results[results.Participant == participant1].Choice.values
]
np.mean(donation)

# %%
np.var(donation)

# %%
results.groupby("Participant").mean()

# %%
first_row.Transcript

# %%
first_row.Choice.payoff_allo

# %%
first_row.Choice.payoff_ego

# %% tags=["hide-input"]

results = pd.read_pickle("../results/dilemma.pickle")
N = len(results)

# %%
import pandas as pd
from llm_cooperation.experiments.ultimatum import *
from llm_cooperation import *

results = pd.read_pickle("../results/ultimatum.pickle")

# %% [raw]
#

# %% [markdown]
# The data consists of a total of

# %% tags=["hide-input"]
print(f"N = {N}")

# %% [markdown]
# cases.  Each case corresponds to a single play of the iterated PD over six rounds.  For each case we record the following fields:

# %% tags=["hide-input"]
results.iloc[0]


# %%
def save_table(df, name, caption):
    df = df.style.format(decimal=".", thousands=",", precision=2)
    df.to_latex(f"../latex/{name}.tex", caption=caption)
    return df


# %% [markdown]
# ### Table 1: Cooperation frequency by group

# %% tags=["hide-input"]
table1 = results.groupby("Group")["Cooperation frequency"].describe().round(2)
save_table(table1, "table1", "Cooperation frequency by group")

# %% [markdown]
# ### Table 2: Cooperation frequency by group/condition

# %% tags=["hide-input"]
table2 = (
    results.groupby(["Group", "Condition"])["Cooperation frequency"].describe().round(2)
)
save_table(table2, "table2", "Cooperation frequency by group/condition")

# %% [markdown]
# ### Table 3: Cooperation frequency by participant

# %% tags=["hide-input"]
table3 = results.groupby("Participant")["Cooperation frequency"].describe().round(2)
save_table(table3, "table3", "Cooperation frequency by participant")

# %% [markdown]
# ### Table 4 - Cooperation frequency by participant and condition

# %% tags=["hide-input"]
table4 = (
    results.groupby(["Participant", "Condition"])["Cooperation frequency"]
    .describe()
    .round(2)
)
save_table(table4, "table4", "Cooperation frequency by participant and condition")

# %% tags=["hide-input"]
import plotly.express as px
import plotly
from IPython.core.display import HTML
from typing import Callable


def graph(fn: Callable, name: str):
    fig = fn()
    fname = f"{name}.html"
    plotly.offline.plot(fig, filename=fname, auto_open=False)
    display(HTML(filename=fname))


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
graph(lambda: px.box(results, x="Group", y="Cooperation frequency"), "figure1")

# %% [markdown]
# ### Figure 2: Cooperation frequency by condition

# %% tags=["hide-input"]
graph(lambda: px.box(results, x="Condition", y="Cooperation frequency"), "figure2")

# %% [markdown]
# ### Figure 3: Cooperation frequency by condition- control group

# %% tags=["hide-input"]
boxplot("Group.Control", "figure3")

# %% [markdown]
# ### Figure 4: Cooperation frequency by condition- altruistic group

# %% tags=["hide-input"]
boxplot("Group.Altruistic", "figure4")

# %% [markdown]
# ### Figure 5: Cooperation frequency by condition- selfish group

# %% tags=["hide-input"]
boxplot("Group.Selfish", "figure5")

# %% [markdown]
# ### Figure 6: Cooperation frequency by condition- mixed group

# %% tags=["hide-input"]
boxplot("Group.Mixed", "figure6")
