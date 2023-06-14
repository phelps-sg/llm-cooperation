import numpy as np
import pytest
from openai_pygenerator import user_message

from llm_cooperation import Choices
from llm_cooperation.dictator import (
    DictatorChoice,
    DictatorEnum,
    all_dictator_choices,
    black,
    blue,
    brown,
    compute_freq_dictator,
    describe_payoffs,
    extract_choice_ultimatum,
    get_prompt_dictator,
    green,
    payoffs_dictator,
    white,
)


@pytest.mark.parametrize(
    "enum, expected_description, expected_payoff_ego, expected_payoff_allo",
    [
        (DictatorEnum.BLACK, "black", 4.0, 0.0),
        (DictatorEnum.BROWN, "brown", 3.0, 1.0),
        (DictatorEnum.GREEN, "green", 2.0, 2.0),
        (DictatorEnum.BLUE, "blue", 1.0, 3.0),
        (DictatorEnum.WHITE, "white", 0.0, 4.0),
    ],
)
def test_dictator_choice(
    enum: DictatorEnum,
    expected_description: str,
    expected_payoff_ego: float,
    expected_payoff_allo,
):
    choice = DictatorChoice(enum)
    assert expected_description in choice.description.lower()
    assert choice.payoff_ego == expected_payoff_ego
    assert choice.payoff_allo == expected_payoff_allo


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Black", black),
        ("Brown", brown),
        ("Green", green),
        ("Blue", blue),
        ("White", white),
    ],
)
def test_extract_choice_ultimatum(text: str, expected_result: DictatorChoice):
    assert (
        extract_choice_ultimatum(user_message(f"'project {text}'")) == expected_result
    )
    assert (
        extract_choice_ultimatum(user_message(f"Project {text.upper()}"))
        == expected_result
    )
    assert (
        extract_choice_ultimatum(user_message(f"Project {text.lower()}"))
        == expected_result
    )


@pytest.mark.parametrize("test_choice", all_dictator_choices)
def test_payoffs_dictator(test_choice: DictatorChoice):
    result = payoffs_dictator(test_choice, green)
    assert result[0] == test_choice.payoff_ego
    assert np.isnan(result[1])


@pytest.mark.parametrize("test_choice", all_dictator_choices)
def test_compute_freq_dictator(test_choice: DictatorChoice):
    result = compute_freq_dictator([Choices(user=green, ai=test_choice)])
    assert result == test_choice.payoff_allo


def test_get_prompt_dictator():
    role_prompt = "test-role"
    result = get_prompt_dictator(1, role_prompt)
    assert role_prompt in result
    for choice in all_dictator_choices:
        assert describe_payoffs(choice) in result
