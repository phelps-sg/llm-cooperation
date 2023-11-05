from typing import List

import pytest
from openai_pygenerator import Completion, user_message

from llm_cooperation.experiments.ultimatum import (
    Accept,
    ProposerChoice,
    Reject,
    extract_choice_ultimatum,
    payoffs_ultimatum,
)
from llm_cooperation.gametypes import alternating
from llm_cooperation.gametypes.repeated import Choices
from tests.test_ultimatum import assistant_message


@pytest.mark.parametrize(
    "i, expected_choices",
    [
        (0, Choices(ai=ProposerChoice(10.0), user=Accept)),  # type: ignore
        (1, Choices(user=ProposerChoice(10.0), ai=Accept)),  # type: ignore
        (2, Choices(ai=ProposerChoice(7.0), user=Reject)),  # type: ignore
        (3, Choices(user=ProposerChoice(10.0), ai=Accept)),  # type: ignore
        (4, Choices(ai=ProposerChoice(5.00), user=Accept)),  # type: ignore
    ],
)
def test_analyse_round(
    i: int,
    expected_choices: Choices,
    alternating_history,
):
    __scores__, choices = alternating.analyse_round(
        i, alternating_history, payoffs_ultimatum, extract_choice_ultimatum
    )
    assert choices == expected_choices


@pytest.fixture
def alternating_history() -> List[Completion]:
    return [
        assistant_message("Propose $10"),
        user_message("Accept,\n Propose $10"),
        assistant_message(
            """I accept your offer.
For the next round I propose $7"""
        ),
        user_message("Reject / Propose $10"),
        assistant_message("I Accept, and then I propose $5"),
        user_message("Accept"),
    ]
