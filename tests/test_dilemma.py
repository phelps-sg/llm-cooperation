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

import re
from typing import List
from unittest.mock import Mock

import pytest
from openai_pygenerator import Completion, logger, user_message
from pytest_lazyfixture import lazy_fixture

from llm_cooperation import DEFAULT_MODEL_SETUP, Group, Participant, Payoffs, exhaustive
from llm_cooperation.experiments import AI_PARTICIPANTS, GROUP_PROMPT_CONDITIONS
from llm_cooperation.experiments.dilemma import (
    CONDITION_LABELS_REVERSED,
    CONDITION_PRONOUN,
    Cooperate,
    Defect,
    DilemmaChoice,
    DilemmaEnum,
    P,
    Pronoun,
    R,
    S,
    T,
    cooperate_label,
    defect_label,
    extract_choice_pd,
    get_choice_template,
    get_participants,
    get_prompt_pd,
    get_pronoun_phrasing,
    move_as_str,
    payoffs_pd,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)
from llm_cooperation.gametypes import simultaneous
from llm_cooperation.gametypes.repeated import GameSetup, play_game
from tests.conftest import modify_condition


@pytest.mark.parametrize(
    "condition",
    [
        lazy_fixture("base_condition"),
        lazy_fixture("with_chain_of_thought"),
    ],
)
def test_get_instruction_prompt(condition: Participant):
    role_prompt = AI_PARTICIPANTS[Group.Control][0]
    prompt = get_prompt_pd(condition)
    logger.debug("prompt = %s", prompt)
    assert "COLOR_COOPERATE" not in prompt
    assert "COLOR_DEFECT" not in prompt
    assert ("Explanation:" in prompt) == condition["chain_of_thought"]
    assert "Choice:" in prompt
    for payoff in [R, S, T, P]:
        assert f"${payoff}.00" in prompt
    assert role_prompt in prompt


@pytest.mark.parametrize(
    ["condition", "expected_regex"],
    [
        (lazy_fixture("base_condition"), r"blue.*green"),
        (lazy_fixture("with_defect_first"), r"green.*blue"),
    ],
)
def test_get_choice_template(condition: Participant, expected_regex: str):
    result = get_choice_template(condition, "blue", "green").lower()
    match = re.search(expected_regex, result)
    assert match is not None


@pytest.mark.parametrize(
    "pronoun, expected",
    [
        (Pronoun.HE.value, "he"),
        (Pronoun.SHE.value, "she"),
        (Pronoun.THEY.value, "they"),
    ],
)
def test_get_pronoun_phrasing(base_condition, pronoun: str, expected: str):
    condition = modify_condition(base_condition, CONDITION_PRONOUN, pronoun)
    assert expected.lower() in get_pronoun_phrasing(condition).lower()


@pytest.mark.parametrize(
    "condition, text, expected_move",
    [
        (lazy_fixture("base_condition"), "project Green", Cooperate),
        (lazy_fixture("with_numbers"), "Choice: 'project one'", Cooperate),
        (lazy_fixture("with_numbers"), "project Two", Defect),
        (lazy_fixture("with_numerals"), "project 1", Cooperate),
        (lazy_fixture("with_numerals"), "project 2", Defect),
        (lazy_fixture("base_condition"), "project Blue", Defect),
        (lazy_fixture("base_condition"), "Project  green", Cooperate),
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
    condition: Participant, text: str, expected_move: DilemmaChoice
):
    move = extract_choice_pd(condition, user_message(text))
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


@pytest.mark.parametrize(
    ["condition", "choice", "expected"],
    [
        (lazy_fixture("base_condition"), DilemmaEnum.C, "Green"),
        (lazy_fixture("base_condition"), DilemmaEnum.D, "Blue"),
        (lazy_fixture("with_numerals"), DilemmaEnum.C, "1"),
        (lazy_fixture("with_numerals"), DilemmaEnum.D, "2"),
        (lazy_fixture("with_numbers"), DilemmaEnum.C, "One"),
        (lazy_fixture("with_numbers"), DilemmaEnum.D, "Two"),
    ],
)
def test_move_as_str(condition: Participant, choice: DilemmaEnum, expected: str):
    assert expected in move_as_str(choice, condition)


@pytest.mark.parametrize(
    ["condition", "expected"],
    [
        (lazy_fixture("base_condition"), "Green"),
        (lazy_fixture("with_numbers"), "One"),
        (lazy_fixture("with_numerals"), "1"),
    ],
)
def test_cooperate_label(condition: Participant, expected: str):
    assert cooperate_label(condition) == expected


@pytest.mark.parametrize(
    ["condition", "expected"],
    [
        (lazy_fixture("base_condition"), "Blue"),
        (lazy_fixture("with_numbers"), "Two"),
        (lazy_fixture("with_numerals"), "2"),
    ],
)
def test_defect_label(condition: Participant, expected: str):
    assert defect_label(condition) == expected


@pytest.mark.parametrize(
    ["condition", "expected"],
    [
        (lazy_fixture("base_condition"), "Blue"),
        (lazy_fixture("with_numerals"), "2"),
        (lazy_fixture("with_numbers"), "Two"),
    ],
)
def test_cooperate_label_reversed(condition: Participant, expected: str):
    condition_reversed = modify_condition(condition, CONDITION_LABELS_REVERSED, True)
    assert cooperate_label(condition_reversed) == expected


def test_get_participants():
    n = 5
    random_participants = get_participants(num_participant_samples=n)
    assert len(random_participants) == n * len(
        list(exhaustive(GROUP_PROMPT_CONDITIONS))
    )
    assert get_participants(n) == random_participants
    factorial_participants = get_participants(num_participant_samples=0)
    assert get_participants(0) == factorial_participants
    assert len(factorial_participants) == 3888


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
            participant=condition,
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
    assert Defect.description(condition) in str(conversation[-1]["content"])
