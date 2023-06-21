import pytest

from llm_cooperation.experiments.ultimatum import (
    Accept,
    ProposerChoice,
    extract_choice_ultimatum,
    payoffs_ultimatum,
)
from llm_cooperation.gametypes import alternating
from llm_cooperation.gametypes.repeated import Choices


@pytest.mark.parametrize(
    "i, expected_choices",
    [
        (0, Choices(ai=ProposerChoice(10.0), user=Accept)),
        (1, Choices(user=ProposerChoice(10.0), ai=Accept)),
    ],
)
def test_analyse_round(
    i: int,
    expected_choices: Choices,
    alternating_history,
):
    _scores, choices = alternating.analyse_round(
        i, alternating_history, payoffs_ultimatum, extract_choice_ultimatum
    )
    assert choices == expected_choices