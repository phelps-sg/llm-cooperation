from typing import List
from unittest.mock import Mock

import pytest
from openai_pygenerator import Completion, logger
from pytest_lazyfixture import lazy_fixture

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
    move_as_str,
    payoffs_pd,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)
from llm_cooperation.gametypes import simultaneous
from llm_cooperation.gametypes.repeated import GameSetup, play_game
from llm_cooperation.main import Settings
from tests.common import make_completion


@pytest.mark.parametrize(
    "condition",
    [
        lazy_fixture("base_condition"),
        lazy_fixture("with_chain_of_thought"),
    ],
)
def test_get_instruction_prompt(condition: Settings):
    role_prompt = "You are a helpful assistant."
    prompt = get_prompt_pd(condition, role_prompt)
    logger.debug("prompt = %s", prompt)
    assert "COLOR_COOPERATE" not in prompt
    assert "COLOR_DEFECT" not in prompt
    assert ("Explanation:" in prompt) == condition["chain_of_thought"]
    assert "Choice:" in prompt
    for payoff in [R, S, T, P]:
        assert f"${payoff}.00" in prompt
    assert role_prompt in prompt


@pytest.mark.parametrize(
    "condition, text, expected_move",
    [
        (lazy_fixture("base_condition"), "project Green", Cooperate),
        (lazy_fixture("base_condition"), "project Blue", Defect),
        (lazy_fixture("base_condition"), "Project green", Cooperate),
        (lazy_fixture("base_condition"), "Project blue", Defect),
        (lazy_fixture("base_condition"), "'project GREEN'", Cooperate),
        (lazy_fixture("base_condition"), "Choice: PROJECT BLUE", Defect),
        (
            lazy_fixture("base_condition"),
            """Explanation: Because project Green is bad for me.
Choice: Project Blue""",
            Defect,
        ),
    ],
)
def test_extract_choice_pd(
    condition: Settings, text: str, expected_move: DilemmaChoice
):
    move = extract_choice_pd(condition, make_completion(text))
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
        (strategy_t4t_cooperate, 6, Defect),
        (strategy_t4t_cooperate, 4, Defect),
        (strategy_t4t_cooperate, 10, Cooperate),
        (strategy_t4t_cooperate, 2, Cooperate),
        (strategy_t4t_defect, 2, Defect),
    ],
)
def test_strategy(strategy, index, expected, conversation, base_condition):
    state = Mock(spec=["messages"])
    state.messages = conversation[:index]
    state.participant_condition = base_condition
    assert strategy(state) == expected


def test_dilemma_choice():
    c1 = DilemmaChoice(DilemmaEnum.C)
    c2 = DilemmaChoice(DilemmaEnum.C)
    d = DilemmaChoice(DilemmaEnum.D)
    assert c1 == c2
    assert c1 != d
    assert Cooperate != Defect


def test_move_as_str(base_condition):
    assert "Green" in move_as_str(DilemmaEnum.C, base_condition)
    assert "Blue" in move_as_str(DilemmaEnum.D, base_condition)


def test_run_repeated_game(mocker, base_condition):
    completions = [
        {"role": "assistant", "content": "project green"},
    ]
    mocker.patch(
        "openai_pygenerator.openai_pygenerator.generate_completions",
        return_value=completions,
    )
    condition = base_condition
    conversation: List[Completion] = list(
        play_game(
            partner_strategy=strategy_defect,
            role_prompt="You are a participant in a psychology experiment",
            participant_condition=condition,
            game_setup=GameSetup(
                num_rounds=3,
                generate_instruction_prompt=get_prompt_pd,
                next_round=simultaneous.next_round,
                analyse_rounds=simultaneous.analyse_rounds,
                payoffs=payoffs_pd,
                extract_choice=extract_choice_pd,
                model_setup=DEFAULT_MODEL_SETUP,
            ),
        )
    )
    assert len(conversation) == 7
    # pylint: disable=unsubscriptable-object
    assert Defect.description(condition) in conversation[-1]["content"]
