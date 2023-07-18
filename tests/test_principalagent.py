import pytest
from openai_pygenerator import user_message

from llm_cooperation.experiments.principalagent import extract_choice_pa


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("Choice: the first", "the first"),
        ("Choice:second", "second"),
        ("choice:  third", "third"),
    ],
)
def test_extract_choice_pa(test_input: str, expected: str):
    result = extract_choice_pa(user_message(test_input))
    assert result.value == expected
