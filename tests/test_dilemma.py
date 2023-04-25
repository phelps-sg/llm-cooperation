from typing import Tuple

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
    strategy_tit_for_tat,
)
from gpt import Completion, Conversation


def make_completion(text: str) -> Completion:
    return {"content": text}


@pytest.fixture
def conversation() -> Conversation:
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
        (strategy_tit_for_tat, 6, Choice.D),
        (strategy_tit_for_tat, 4, Choice.C),
    ],
)
def test_strategy(strategy, index, expected, conversation):
    assert strategy(conversation[:index]) == expected


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
