from unittest.mock import Mock

import numpy as np
import pandas as pd
from openai_pygenerator import content, user_message

from llm_cooperation import Choice, Group
from llm_cooperation.oneshot import (
    analyse,
    compute_scores,
    generate_samples,
    play_game,
    run_experiment,
)


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


def test_play_game(mocker):
    test_prompt = "test-prompt"
    mock_completion = {"role": "assistant", "content": "test-response"}
    mocker.patch(
        "openai_pygenerator.openai_pygenerator.generate_completions",
        return_value=[mock_completion],
    )
    mock_generate = Mock()
    mock_generate.return_value = test_prompt
    result = play_game(
        role_prompt="test-role-prompt", generate_instruction_prompt=mock_generate
    )
    print(result)
    assert len(result) == 2
    assert result[1] == mock_completion
    assert content(result[0]) == test_prompt


def test_analyse(mocker):
    mock_choice = Mock(spec=Choice)
    test_score = 1.0
    mocker.patch(
        "llm_cooperation.oneshot.compute_scores", return_value=(test_score, mock_choice)
    )

    test_freq = 2.0
    mock_compute_freq = Mock()
    mock_compute_freq.return_value = test_freq

    test_messages = [f"test{i}" for i in range(3)]

    result = analyse(
        conversation=[user_message(c) for c in test_messages],
        payoffs=Mock(),
        extract_choice=Mock(),
        compute_freq=mock_compute_freq,
    )
    assert result == (test_score, test_freq, mock_choice, test_messages)

    mock_compute_freq_with_ex = Mock()
    test_err_message = "test exception"
    mock_compute_freq_with_ex.side_effect = ValueError(test_err_message)
    err_result = analyse(
        conversation=[],
        payoffs=Mock(),
        extract_choice=Mock(),
        compute_freq=mock_compute_freq_with_ex,
    )
    assert len(err_result) == 4
    assert np.isnan(err_result[1])
    assert err_result[2] is None
    assert test_err_message in err_result[3]
