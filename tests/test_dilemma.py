from typing import List
from unittest.mock import Mock

import pytest
from openai_pygenerator import Completion

from llm_cooperation import DEFAULT_MODEL_SETUP, Payoffs
from llm_cooperation.experiments.dilemma import (
    Cooperate,
    Defect,
    DilemmaChoice,
    DilemmaEnum,
    P,
    R,
    S,
    T,
    extract_choice_pd,
    get_prompt_pd,
    payoffs_pd,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)
from llm_cooperation.gametypes import simultaneous
from llm_cooperation.gametypes.repeated import GameSetup, play_game
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
        ("Choice: 'project green'", Cooperate),
        (
            """Explanation: Because project green is bad for me.
Choice: Project BLUE""",
            Defect,
        ),
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
    expected_payoffs: Payoffs,
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
    state = Mock(spec=["messages"])
    state.messages = conversation[:index]
    assert strategy(state) == expected


def test_dilemma_choice():
    c1 = DilemmaChoice(DilemmaEnum.C)
    c2 = DilemmaChoice(DilemmaEnum.C)
    d = DilemmaChoice(DilemmaEnum.D)
    assert c1 == c2
    assert c1 != d
    assert Cooperate != Defect


def test_run_repeated_game(mocker):
    completions = [
        {"role": "assistant", "content": "project green"},
    ]
    mocker.patch(
        "openai_pygenerator.openai_pygenerator.generate_completions",
        return_value=completions,
    )
    conversation: List[Completion] = list(
        play_game(
            partner_strategy=strategy_defect,
            role_prompt="You are a participant in a psychology experiment",
            participant_condition=False,
            game_setup=GameSetup(
                num_rounds=3,
                generate_instruction_prompt=get_prompt_pd,
                next_round=simultaneous.next_round,
                rounds=simultaneous.rounds_setup,
                payoffs=payoffs_pd,
                extract_choice=extract_choice_pd,
                model_setup=DEFAULT_MODEL_SETUP,
            ),
        )
    )
    assert len(conversation) == 7
    # pylint: disable=unsubscriptable-object
    assert Defect.description in conversation[-1]["content"]
