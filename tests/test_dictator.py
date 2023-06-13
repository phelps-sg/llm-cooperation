import pytest

from llm_cooperation.dictator import (
    DictatorChoice,
    DictatorEnum,
    black,
    blue,
    brown,
    describe_payoffs,
    get_prompt_dictator,
    green,
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


@pytest.mark.parametrize("choice", [black, brown, green, blue, white])
def test_get_prompt_dictator(choice: DictatorChoice):
    role_prompt = "test-role"
    result = get_prompt_dictator(role_prompt)
    assert role_prompt in result
    assert describe_payoffs(choice) in result
