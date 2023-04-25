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
    move_as_str, Choices,
)


def test_get_prompt():
    prompt = get_prompt(6)
    assert "you will do so 6 times." in prompt


@pytest.mark.parametrize(
    "completion, expected_move",
    [
        ("project green", Choice.C),
        ("project blue", Choice.D),
        ("Project GREEN", Choice.C),
        ("Project BLUE", Choice.D),
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


def test_compute_scores():
    conversation = [
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

    scores, moves = compute_scores(conversation)
    assert scores == Scores(ai=T + S + P + T, user=S + T + P + S)
    assert moves == [
        Choices(Choice.D, Choice.C),
        Choices(Choice.C, Choice.D),
        Choices(Choice.D, Choice.D),
        Choices(Choice.C, Choice.D),
    ]


@pytest.fixture
def mock_generate_completions(mocker):
    return mocker.patch("gpt.generate_completions")


@pytest.fixture
def mock_environment(mocker):
    return mocker.patch("os.environ")


def test_run_prisoners_dilemma(mocker):
    completions = [
        {"role": "assistant", "content": "project green"},
    ]
    mocker.patch("gpt.generate_completions", return_value=completions)
    conversation = list(run_prisoners_dilemma(num_rounds=3))
    assert len(conversation) == 8
    assert move_as_str(Choice.D) in conversation[-1]["content"]
