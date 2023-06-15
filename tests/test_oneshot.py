from unittest.mock import Mock

import pandas as pd

from llm_cooperation import Choice, Group
from llm_cooperation.oneshot import compute_scores, generate_samples, run_experiment


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


def test_generate_samples(mocker):
    mock_result_row = (0.5, 0.2, None, [])
    mocker.patch("llm_cooperation.oneshot.analyse", return_value=mock_result_row)
    mocker.patch("llm_cooperation.oneshot.play_game", return_value=[])
    test_n = 3
    result = list(
        generate_samples(
            prompt="test-prompt",
            num_samples=test_n,
            generate_instruction_prompt=Mock(),
            payoffs=Mock(),
            extract_choice=Mock(),
            compute_freq=Mock(),
        )
    )
    assert len(result) == test_n
    assert result == [mock_result_row for _i in range(3)]


def test_compute_scores():
    mock_choice = Mock(spec=Choice)
    mock_payoff = 0.5
    result = compute_scores(
        conversation=["prompt", "answer"],
        payoffs=lambda _: mock_payoff,
        extract_choice=lambda _: mock_choice,
    )
    assert result == (mock_payoff, mock_choice)
