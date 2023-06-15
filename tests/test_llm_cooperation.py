import os
from unittest.mock import MagicMock

import pandas as pd

from llm_cooperation import Results, run_and_record_experiment


def test_run_and_record_experiment():
    df_mock = MagicMock(spec=pd.DataFrame)
    results_mock = MagicMock(spec=Results)
    results_mock.to_df = MagicMock(return_value=df_mock)
    name = "test_experiment"
    run_and_record_experiment(name, lambda: results_mock)
    filename = os.path.join("results", f"{name}.pickle")
    df_mock.to_pickle.assert_called_once_with(filename)
