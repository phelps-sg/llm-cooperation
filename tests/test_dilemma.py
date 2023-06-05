from typing import Tuple
from unittest.mock import Mock

import pytest

from llm_cooperation import Group, run_experiment
from llm_cooperation.dilemma import (
    Cooperate,
    Defect,
    DilemmaChoice,
    DilemmaEnum,
    P,
    R,
    S,
    T,
    compute_freq_pd,
    extract_choice_pd,
    get_prompt_pd,
    payoffs_pd,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)
from tests.common import make_completion


def test_get_instruction_prompt():
    rounds = 6
    role_prompt = "You are a helpful assistant."
    prompt = get_prompt_pd(rounds, role_prompt)
    assert f"{rounds} rounds" in prompt
    for payoff in [R, S, T, P]:
        assert f"${payoff}.00" in prompt
    assert role_prompt in prompt


@pytest.mark.parametrize(
    "text, expected_move",
    [
        ("project green", Cooperate),
        ("project blue", Defect),
        ("Project GREEN", Cooperate),
        ("Project BLUE", Defect),
        ("'project green'", Cooperate),
    ],
)
def test_extract_choice_pd(text: str, expected_move: DilemmaChoice):
    move = extract_choice_pd(make_completion(text))
    assert move == expected_move


@pytest.mark.parametrize(
    "user_choice, partner_choice, expected_payoffs",
    [
        (Defect, Cooperate, (T, S)),
        (Cooperate, Cooperate, (R, R)),
        (Defect, Defect, (P, P)),
        (Cooperate, Defect, (S, T)),
    ],
)
def test_payoffs(
    user_choice: DilemmaChoice,
    partner_choice: DilemmaChoice,
    expected_payoffs: Tuple[int, int],
):
    user_payoff, partner_payoff = payoffs_pd(user_choice, partner_choice)
    assert (user_payoff, partner_payoff) == expected_payoffs


@pytest.mark.parametrize(
    "strategy, index, expected",
    [
        (strategy_cooperate, 5, Cooperate),
        (strategy_cooperate, 3, Cooperate),
        (strategy_defect, 5, Defect),
        (strategy_defect, 3, Defect),
        (strategy_t4t_cooperate, 5, Defect),
        (strategy_t4t_cooperate, 3, Cooperate),
        (strategy_t4t_cooperate, 2, Cooperate),
        (strategy_t4t_defect, 2, Defect),
    ],
)
def test_strategy(strategy, index, expected, conversation):
    assert strategy(conversation[:index]) == expected


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch("llm_cooperation.run_sample")
    samples = [
        (5, 0.5, [Cooperate], ["project green"]),
        (3, 0.7, [Defect], ["project blue"]),
        (6, 0.6, [Defect], ["project blue"]),
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

    result = list(
        run_experiment(
            ai_participants,
            user_conditions,
            num_rounds=6,
            num_samples=len(samples),
            generate_instruction_prompt=get_prompt_pd,
            payoffs=payoffs_pd,
            extract_choice=extract_choice_pd,
            compute_freq=compute_freq_pd,
        )
    )
    assert len(result) == len(samples) * len(user_conditions) * 3
    assert mock_run_sample.call_count == len(samples) * len(user_conditions)


def test_dilemma_choice():
    c1 = DilemmaChoice(DilemmaEnum.C)
    c2 = DilemmaChoice(DilemmaEnum.C)
    d = DilemmaChoice(DilemmaEnum.D)
    assert c1 == c2
    assert c1 != d
    assert Cooperate != Defect
