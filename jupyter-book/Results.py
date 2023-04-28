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

# %%
import pandas as pd
from dilemma import Group
# %%
results = pd.read_pickle("../results.pickle")
results.columns = ["score (mean)", "score (std)", "cooperation frequency (mean)", "cooperation frequency (std)", "N"]
results.round(decimals=2)

# %%
results.index = results.index.map(lambda x: (str(x[0]), x[1], x[2]))

# %%
results.groupby(level=2).describe()
