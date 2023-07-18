import pytest
from openai_pygenerator import user_message

from llm_cooperation.experiments.principalagent import extract_choice_pa


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("Choice: id 1", 1),
        ("Choice:2", 2),
        ("choice:  product id: 35", 35),
    ],
)
def test_extract_choice_pa(test_input: str, expected: int):
    result = extract_choice_pa(user_message(test_input))
    assert result.value == expected
