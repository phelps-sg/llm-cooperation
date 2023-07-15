import os
from unittest.mock import MagicMock

import pandas as pd

from llm_cooperation import DEFAULT_MODEL_SETUP, ModelSetup, Results
from llm_cooperation.experiments import (
    DEFAULT_SAMPLE_SIZE,
    create_results_dir,
    run_and_record_experiment,
)


def test_create_results_dir(mocker):
    model_setup = ModelSetup(model="test-model", temperature=0.2, max_tokens=100)
    create_dir = mocker.patch("llm_cooperation.experiments.create_dir")
    create_results_dir(model_setup)
    create_dir.assert_called_once()
    first_arg = create_dir.call_args[0][0]
    assert "test-model" in first_arg
    assert "0.2" in first_arg


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
    run_and_record_experiment(
        name,
        run=lambda _setup, _n: results_mock,
        model_setup=DEFAULT_MODEL_SETUP,
        sample_size=DEFAULT_SAMPLE_SIZE,
    )

    pickle_filename = os.path.join(results_dir, f"{name}.pickle")
    df_mock.to_pickle.assert_called_once_with(pickle_filename)

    csv_filename = os.path.join(results_dir, f"{name}.csv")
    df_mock.to_csv.assert_called_once_with(csv_filename)
