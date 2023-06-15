from unittest.mock import Mock

import pandas as pd

from llm_cooperation import Choice, Group
from llm_cooperation.oneshot import run_experiment


def test_run_experiment(mocker):
    mock_choice = Mock(spec=Choice)

    mock_run_sample = mocker.patch("llm_cooperation.oneshot.generate_samples")
    samples = [
        (5, 0.5, mock_choice, ["project green"]),
        (3, 0.7, mock_choice, ["project blue"]),
        (6, 0.6, mock_choice, ["project blue"]),
    ]
    mock_run_sample.return_value = samples

    ai_participants = {
        Group.Altruistic: ["Participant 1", "Participant 2"],
        Group.Control: ["Participant 3"],
    }

    result: pd.DataFrame = run_experiment(
        ai_participants,
        num_samples=len(samples),
        generate_instruction_prompt=Mock(),
        payoffs=Mock(),
        extract_choice=Mock(),
        compute_freq=Mock(),
    ).to_df()
    assert len(result) == 3 * len(samples)
    assert mock_run_sample.call_count == len(samples)
