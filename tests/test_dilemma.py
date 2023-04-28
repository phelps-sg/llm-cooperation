from typing import Iterable, List, Tuple
from unittest.mock import Mock

import pytest

from dilemma import (
    PAYOFFS_PD,
    Choice,
    Choices,
    Group,
    P,
    R,
    ResultRow,
    S,
    Scores,
    T,
    compute_scores,
    extract_choice,
    get_prompt,
    move_as_str,
    payoffs,
    results_to_df,
    run_experiment,
    run_prisoners_dilemma,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)
from gpt import Completion


def make_completion(text: str) -> Completion:
    return {"content": text}


@pytest.fixture
def cooperate_choices() -> List[Choices]:
    return [Choices(Choice.C, Choice.C) for _i in range(3)]


@pytest.fixture
def defect_choices() -> List[Choices]:
    return [Choices(Choice.D, Choice.D) for _i in range(3)]


@pytest.fixture
def results(cooperate_choices, defect_choices) -> Iterable[ResultRow]:
    return iter(
        [
            (
                Group.Altruistic,
                "You are altruistic",
                "unconditional cooperate",
                30,
                0.2,
                cooperate_choices,
                ["project green", "project green", "project green"],
            ),
            (
                Group.Selfish,
                "You are selfish",
                "unconditional cooperate",
                60,
                0.5,
                defect_choices,
                ["project blue", "project blue", "project blue"],
            ),
        ]
    )


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


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch("dilemma.run_sample")
    samples = [
        (5, 0.5, [Choice.C], ["project green"]),
        (3, 0.7, [Choice.D], ["project blue"]),
        (6, 0.6, [Choice.D], ["project blue"]),
    ]
    mock_run_sample.return_value = samples

    ai_participants = {
        Group.Altruistic: ["Participant 1", "Participant 2"],
        Group.Control: ["Participant 3"],
    }
    user_conditions = {
        "strategy_A": Mock(),
        "strategy_B": Mock(),
    }

    result = list(run_experiment(ai_participants, user_conditions))
    assert len(result) == len(samples) * len(user_conditions) * 3
    assert mock_run_sample.call_count == len(samples) * len(user_conditions)


def test_results_to_df(results: Iterable[ResultRow]):
    df = results_to_df(results)
    assert len(df.columns) == 7
    assert len(df) == 2
    assert df["Group"][0] == str(Group.Altruistic)
    assert df["Group"][1] == str(Group.Selfish)


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
