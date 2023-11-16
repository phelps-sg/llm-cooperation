#  MIT License
#
#  Copyright (c) 2023 Steve Phelps
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

from typing import Iterable
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from openai_pygenerator import content, user_message

from llm_cooperation import DEFAULT_MODEL_SETUP, Choice, Grid, Participant
from llm_cooperation.experiments.dictator import CONDITION_ROLE
from llm_cooperation.gametypes.oneshot import (
    OneShotResults,
    ResultSingleShotGame,
    analyse,
    compute_scores,
    generate_replications,
    play_game,
    run_experiment,
)


def test_run_experiment(mocker):
    mock_choice = Mock(spec=Choice)

    mock_run_sample = mocker.patch(
        "llm_cooperation.gametypes.oneshot.generate_replications"
    )
    samples = [
        (5, 0.5, mock_choice, ["project green"]),
        (3, 0.7, mock_choice, ["project blue"]),
        (6, 0.6, mock_choice, ["project blue"]),
    ]
    mock_run_sample.return_value = samples
    n = 5
    result: pd.DataFrame = run_experiment(
        participants=(
            (Participant({CONDITION_ROLE: f"Participant {i}"}) for i in range(n))
        ),
        num_replications=3,
        generate_instruction_prompt=Mock(),
        payoffs=Mock(),
        extract_choice=Mock(),
        compute_freq=Mock(),
        model_setup=DEFAULT_MODEL_SETUP,
    ).to_df()
    assert len(result) == 5 * len(samples)
    assert mock_run_sample.call_count == n


def test_generate_samples(mocker):
    mock_result_row = (0.5, 0.2, None, [])
    mocker.patch(
        "llm_cooperation.gametypes.oneshot.analyse", return_value=mock_result_row
    )
    mocker.patch("llm_cooperation.gametypes.oneshot.play_game", return_value=[])
    test_n = 3
    result = list(
        generate_replications(
            num_replications=test_n,
            generate_instruction_prompt=Mock(),
            payoffs=Mock(),
            extract_choice=Mock(),
            compute_freq=Mock(),
            model_setup=DEFAULT_MODEL_SETUP,
            participant=Participant({CONDITION_ROLE: "Group 1"}),
        )
    )
    assert len(result) == test_n
    assert result == [mock_result_row for __i__ in range(3)]


def test_compute_scores(base_condition: Participant):
    mock_payoff = 0.5
    mock_choice = Mock(spec=Choice)
    result = compute_scores(
        conversation=[user_message("prompt"), user_message("answer")],
        payoffs=lambda _: mock_payoff,
        extract_choice=lambda __condition__, __completion__: mock_choice,
        participant_condition=base_condition,
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
        participant=Participant({CONDITION_ROLE: "test"}),
        generate_instruction_prompt=mock_generate,
        model_setup=DEFAULT_MODEL_SETUP,
    )
    assert len(result) == 2
    assert result[1] == mock_completion
    assert content(result[0]) == test_prompt


def test_analyse(mocker):
    mock_choice = Mock(spec=Choice)
    test_score = 1.0
    mocker.patch(
        "llm_cooperation.gametypes.oneshot.compute_scores",
        return_value=(test_score, mock_choice),
    )

    test_freq = 2.0
    mock_compute_freq = Mock()
    mock_compute_freq.return_value = test_freq

    test_messages = [f"test{i}" for i in range(3)]
    test_condition: Participant = Participant({"chain_of_thought": True})

    result = analyse(
        conversation=[user_message(c) for c in test_messages],
        payoffs=Mock(),
        extract_choice=Mock(),
        compute_freq=mock_compute_freq,
        participant_condition=test_condition,
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
        participant_condition=test_condition,
    )
    assert len(err_result) == 4
    assert np.isnan(err_result[1])
    assert err_result[2] is None
    assert test_err_message in err_result[3]


def test_results_to_df(results: Iterable[ResultSingleShotGame]):
    df = OneShotResults(results).to_df()
    assert len(df.columns) == 7
    # pylint: disable=R0801
    assert len(df) == 2
    assert df["Participant"].iloc[0]["condition"]
    assert not df["Participant"].iloc[1]["condition"]


@pytest.fixture
def results() -> Iterable[ResultSingleShotGame]:
    return iter(
        [
            (
                {"condition": True},
                30.0,
                0.2,
                Mock(spec=Choice),
                ["project green"],
                "gpt-turbo-3.5",
                0.2,
            ),
            (
                {"condition": False},
                60.0,
                0.5,
                Mock(spec=Choice),
                ["project blue"],
                "gpt-4",
                0.1,
            ),
        ]
    )  # type: ignore


@pytest.fixture
def participant_conditions() -> Grid:
    return {"+": [True, False]}


@pytest.fixture
def base_condition() -> Participant:
    return Participant(dict())
