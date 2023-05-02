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

results = pd.read_pickle("../results.pickle")

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

# %%
graph(lambda: px.box(results, x="Group", y="Cooperation frequency"), "figure1")

# %% [markdown]
# ### Figure 2: Cooperation frequency by condition

# %%
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
