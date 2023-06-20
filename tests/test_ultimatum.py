from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
from openai_pygenerator import Completion, content, user_message

from llm_cooperation import Choice, Payoffs, amount_as_str
from llm_cooperation.repeated import Choices, analyse_round
from llm_cooperation.ultimatum import (
    MAX_AMOUNT,
    Accept,
    ProposerChoice,
    Reject,
    ResponderChoice,
    ResponderEnum,
    UltimatumChoice,
    amount_from_str,
    compute_freq_ultimatum,
    extract_choice_ultimatum,
    get_prompt_ultimatum,
    next_round_ultimatum,
    payoffs_ultimatum,
    strategy_cooperate,
)
from tests.common import make_completion


@pytest.mark.parametrize(
    "amount, expected", [(10, "$10.00"), (0, "$0.00"), (0.5, "$0.50")]
)
def test_amount_as_str(amount: float, expected: str):
    assert amount_as_str(amount) == expected


def test_get_instruction_prompt():
    rounds = 6
    role_prompt = "You are a helpful assistant."
    prompt = get_prompt_ultimatum(rounds, role_prompt)
    assert f"{rounds} rounds" in prompt
    assert role_prompt in prompt


@pytest.mark.parametrize(
    "amount_str, expected",
    [
        ("$10.00", 10.0),
        ("$3", 3.0),
        ("$3.50", 3.50),
        ("$0.00", 0.0),
        ("$0", 0.0),
    ],
)
def test_amount_from_str(amount_str: str, expected: float):
    assert amount_from_str(amount_str) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("accept", Accept),
        ("reject", Reject),
        ("ACCEPT", Accept),
        ("REJECT", Reject),
        ("Accept", Accept),
        ("Reject", Reject),
        ("Rejected", Reject),
        (" Rejected", Reject),
        ("Accepted", Accept),
        (" Rejected", Reject),
        ("'Rejected'", Reject),
        (" 'Rejected'", Reject),
        ("$5.00", ProposerChoice(5.0)),
        ("$1.00", ProposerChoice(1.0)),
        (" $1.00", ProposerChoice(1.0)),
        ("$1.00 ", ProposerChoice(1.0)),
        ("$0", ProposerChoice(0)),
    ],
)
def test_extract_choice(text: str, expected: UltimatumChoice):
    assert extract_choice_ultimatum(make_completion(text)) == expected


@pytest.mark.parametrize(
    "player1, player2, expected",
    [
        (ProposerChoice(10.0), Reject, (0.0, 0.0)),
        (ProposerChoice(10.0), Accept, (0.0, 10.0)),
        (ProposerChoice(3.0), Accept, (7.0, 3.0)),
        (ProposerChoice(3.0), Reject, (0.0, 0.0)),
        (ProposerChoice(0.0), Accept, (10.0, 0.0)),
        (ProposerChoice(0.0), Reject, (0.0, 0.0)),
        (Reject, ProposerChoice(10.0), (0.0, 0.0)),
        (Accept, ProposerChoice(10.0), (10.0, 0.0)),
        (Accept, ProposerChoice(3.0), (3.0, 7.0)),
        (Reject, ProposerChoice(3.0), (0.0, 0.0)),
        (Accept, ProposerChoice(0.0), (0.0, 10.0)),
        (Reject, ProposerChoice(0.0), (0.0, 0.0)),
    ],
)
def test_payoffs_ultimatum(
    player1: UltimatumChoice, player2: UltimatumChoice, expected: Payoffs
):
    assert payoffs_ultimatum(player1, player2) == expected


@pytest.mark.parametrize(
    "choices, expected",
    [
        ([ProposerChoice(10.0), Accept, ProposerChoice(5.0), Reject], 0.75),
        ([ProposerChoice(10.0), Accept, ProposerChoice(10.0), Reject], 1.0),
        ([ProposerChoice(0.0), Accept, ProposerChoice(0.0), Reject], 0.0),
    ],
)
def test_compute_freq_ultimatum(choices: List[Choices], expected: float):
    assert np.isclose(compute_freq_ultimatum(choices), expected)


@pytest.mark.parametrize(
    "last_response, expected",
    [
        ("accept", ProposerChoice(MAX_AMOUNT)),
        ("reject", ProposerChoice(MAX_AMOUNT)),
        ("$5.00", Accept),
        ("$10.00", Accept),
        ("$0.00", Accept),
    ],
)
def test_cooperate(last_response: str, expected: Choice):
    assert strategy_cooperate([make_completion(last_response)]) == expected


def test_ultimatum_choice():
    assert ProposerChoice(10.0) == ProposerChoice(10.0)
    assert ProposerChoice(0.0) != ProposerChoice(5.0)
    assert ProposerChoice(10.0) != Accept
    assert Accept != Reject
    assert Accept == ResponderChoice(ResponderEnum.Accept)
    assert Reject == ResponderChoice(ResponderEnum.Reject)


def assistant_message(description):
    return {"role": "assistant", "content": description}


def test_next_round_ultimatum(response, proposal):
    initial_history = [assistant_message(proposal.description)]
    test_strategy = Mock()
    test_strategy.side_effect = [response, proposal]
    result = next_round_ultimatum(test_strategy, initial_history)
    assert response.description in content(result[0])
    assert proposal.description in content(result[1])


@pytest.mark.parametrize(
    "user_class, ai_class",
    [
        (ResponderChoice, ProposerChoice),
    ],
)
def test_analyse_round(user_class: type, ai_class: type, history: List[Completion]):
    _scores, choices = analyse_round(
        0, history, payoffs_ultimatum, extract_choice_ultimatum
    )
    assert isinstance(choices.user, user_class)
    assert isinstance(choices.ai, ai_class)


@pytest.fixture
def history(response, proposal) -> List[Completion]:
    return [
        assistant_message(proposal.description),
        user_message(response.description),
        user_message(proposal.description),
        assistant_message(response.description),
        user_message(proposal.description),
        assistant_message(response.description),
        assistant_message(proposal.description),
        user_message(response.description),
        user_message(proposal.description),
    ]


@pytest.fixture
def response() -> UltimatumChoice:
    return Accept


@pytest.fixture
def proposal() -> UltimatumChoice:
    return ProposerChoice(MAX_AMOUNT)
