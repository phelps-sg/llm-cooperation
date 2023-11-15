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
