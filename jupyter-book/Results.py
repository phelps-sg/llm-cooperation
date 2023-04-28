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

# %%
results = pd.read_pickle("../results.pickle")
results.round(decimals=2)

# %%
results.groupby("Participant").describe()

# %%
results.groupby("Group").describe()

# %%
results.groupby(["Group", "Condition"]).describe()

# %%
import matplotlib.pyplot as plt

_ = results.groupby("Group").boxplot(showfliers=False, figsize=(20, 12), subplots=False)
