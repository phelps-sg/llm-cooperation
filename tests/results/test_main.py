from unittest.mock import call

from llm_cooperation.main import experiments, run_all


def test_run_all(mocker):
    run_and_record = mocker.patch(
        "llm_cooperation.main.run_and_record_experiment", return_value=None
    )
    run_all()
    run_and_record.assert_has_calls(
        [call(experiment, runnable) for experiment, runnable in experiments.items()]
    )
