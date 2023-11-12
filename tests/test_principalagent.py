import pytest
from openai_pygenerator import user_message

from llm_cooperation import Settings
from llm_cooperation.experiments.principalagent import (
    PARTICIPANT_SHELL,
    extract_choice_pa,
    get_prompt_principal_agent,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("Choice: id 1", 1),
        ("Choice:2", 2),
        ("choice:  product id: 35", 35),
    ],
)
def test_extract_choice_pa(test_input: str, expected: int):
    result = extract_choice_pa(dict(), user_message(test_input))
    assert result.value == expected


def test_get_prompt_principal_agent():
    test_condition: Settings = {"shared_with_principal": True, "shared_with_user": True}
    test_role = PARTICIPANT_SHELL
    result = get_prompt_principal_agent(test_condition, test_role)
    assert test_role.principal in result
    assert test_role.simulacrum in result
    assert test_role.query_results in result
    assert test_role.customer in result
    assert "film" not in result.lower()
