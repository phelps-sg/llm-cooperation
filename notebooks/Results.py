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

# %%
import pandas as pd
# %%
results = pd.read_pickle("../results.pickle")
results.columns = ["score (mean)", "score (std)", "cooperation frequency (mean)", "cooperation frequency (std)", "N"]
results.round(decimals=2)
