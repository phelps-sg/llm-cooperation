import os
from unittest.mock import MagicMock

import pandas as pd

from llm_cooperation import Results
from llm_cooperation.experiments import run_and_record_experiment


def test_run_and_record_experiment(mocker):
    df_mock = MagicMock(spec=pd.DataFrame)
    results_mock = MagicMock(spec=Results)
    results_mock.to_df = MagicMock(return_value=df_mock)
    results_dir = "test-results-dir"
    create_results_dir_mock = mocker.patch(
        "llm_cooperation.experiments.create_results_dir"
    )
    create_results_dir_mock.return_value = results_dir
    name = "test_experiment"
    run_and_record_experiment(name, experiment=lambda _setup: results_mock)
    filename = os.path.join(results_dir, f"{name}.pickle")
    df_mock.to_pickle.assert_called_once_with(filename)
