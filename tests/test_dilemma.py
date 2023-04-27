from typing import Tuple, List
from unittest.mock import Mock

import numpy as np
import pytest

from dilemma import (
    get_prompt,
    extract_choice,
    compute_scores,
    Choice,
    Scores,
    run_prisoners_dilemma,
    payoffs,
    PAYOFFS_PD,
    T,
    S,
    P,
    R,
    move_as_str,
    Choices,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    mean,
    Results,
    results_as_df,
    print_report,
    run_experiment,
    strategy_t4t_defect,
    Group,
)
from gpt import Completion


def make_completion(text: str) -> Completion:
    return {"content": text}


@pytest.fixture()
def results() -> Results:
    return {
        (Group.Selfish, "selfish prompt", "tit-for-tat"): (1.0, 2.0, 3.0, 4.0, 10),
        (Group.Altruistic, "altruistic prompt", "defect"): (
            10.0,
            20.0,
            30.0,
            40.0,
            100,
        ),
    }


@pytest.fixture
def conversation() -> List[Completion]:
    return [
        {"system": "system prompt"},
        {"user": "scenario prompt"},
        {"role": "assistant", "content": "project green"},
        {"role": "user", "content": "project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project green"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project green"},
    ]


def test_get_prompt():
    prompt = get_prompt(6)
    assert "you will do so 6 times." in prompt


@pytest.mark.parametrize(
    "completion, expected_move",
    [
        (make_completion("project green"), Choice.C),
        (make_completion("project blue"), Choice.D),
        (make_completion("Project GREEN"), Choice.C),
        (make_completion("Project BLUE"), Choice.D),
    ],
)
def test_extract_choice(completion, expected_move):
    move = extract_choice(completion)
    assert move == expected_move


@pytest.mark.parametrize(
    "user_choice, partner_choice, expected_payoffs",
    [
        (Choice.D, Choice.C, (T, S)),
        (Choice.C, Choice.C, (R, R)),
        (Choice.D, Choice.D, (P, P)),
        (Choice.C, Choice.D, (S, T)),
    ],
)
def test_payoffs(
    user_choice: Choice, partner_choice: Choice, expected_payoffs: Tuple[int, int]
):
    payoff_matrix = PAYOFFS_PD
    user_payoff, partner_payoff = payoffs(user_choice, partner_choice, payoff_matrix)
    assert (user_payoff, partner_payoff) == expected_payoffs


@pytest.mark.parametrize(
    "strategy, index, expected",
    [
        (strategy_cooperate, 6, Choice.C),
        (strategy_cooperate, 4, Choice.C),
        (strategy_defect, 6, Choice.D),
        (strategy_defect, 4, Choice.D),
        (strategy_t4t_cooperate, 6, Choice.D),
        (strategy_t4t_cooperate, 4, Choice.C),
        (strategy_t4t_cooperate, 3, Choice.C),
        (strategy_t4t_defect, 3, Choice.D),
    ],
)
def test_strategy(strategy, index, expected, conversation):
    assert strategy(conversation[:index]) == expected


def test_results_as_df(results):
    df = results_as_df(results)
    assert len(df) == 2
    assert len(df.columns) == 5


def test_print_report(results):
    print_report(results)


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch("dilemma.run_sample")
    mock_run_sample.return_value = [(5, 0.5), (3, 0.7), (6, 0.6)]

    mock_print_report = mocker.patch("dilemma.print_report")

    ai_participants = {
        Group.Altruistic: ["Participant 1", "Participant 2"],
        Group.Control: ["Participant 3"],
    }
    user_conditions = {
        "strategy_A": Mock(),
        "strategy_B": Mock(),
    }

    run_experiment(ai_participants, user_conditions)

    assert mock_run_sample.call_count == 3 * len(user_conditions)
    mock_print_report.assert_called_once()


def test_mean():
    # noinspection PyTypeChecker
    assert mean([2, 3, np.nan, 2, 3.0]) == 2.5


def test_compute_scores(conversation):
    scores, moves = compute_scores(conversation)
    assert scores == Scores(ai=T + S + P + T, user=S + T + P + S)
    assert moves == [
        Choices(Choice.D, Choice.C),
        Choices(Choice.C, Choice.D),
        Choices(Choice.D, Choice.D),
        Choices(Choice.C, Choice.D),
    ]


def test_run_prisoners_dilemma(mocker):
    completions = [
        {"role": "assistant", "content": "project green"},
    ]
    mocker.patch("gpt.generate_completions", return_value=completions)
    conversation = list(run_prisoners_dilemma(num_rounds=3))
    assert len(conversation) == 8
    assert move_as_str(Choice.D) in conversation[-1]["content"]
