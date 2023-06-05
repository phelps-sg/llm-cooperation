from typing import List

import numpy as np
import pytest

from llm_cooperation import Choice, Choices, Payoffs
from llm_cooperation.ultimatum import (
    Accept,
    ProposerChoice,
    Reject,
    UltimatumChoice,
    amount_as_str,
    amount_from_str,
    compute_freq_ultimatum,
    extract_choice_ultimatum,
    get_prompt_ultimatum,
    payoffs_ultimatum,
)
from tests.common import make_completion


@pytest.mark.parametrize(
    "amount, expected", [(10, "$10.00"), (0, "$0.00"), (0.5, "$0.50")]
)
def test_amount_as_str(amount: float, expected: str):
    assert amount_as_str(amount) == expected


def test_get_instruction_prompt():
    rounds = 6
    prompt = get_prompt_ultimatum(rounds)
    assert f"{rounds} rounds" in prompt


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
def test_payoffs_ultimatum(player1: Choice, player2: Choice, expected: Payoffs):
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
