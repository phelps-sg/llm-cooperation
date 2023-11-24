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
from pytest_lazyfixture import lazy_fixture

from llm_cooperation import Participant
from llm_cooperation.experiments.principalagent import (
    ATTRIBUTE_CUSTOMER,
    ATTRIBUTE_PRINCIPAL,
    ATTRIBUTE_SIMULACRUM,
    PARTICIPANT_SHELL,
    extract_choice_pa,
    get_prompt_principal_agent,
    query_results,
)


@pytest.mark.parametrize(
    "condition, test_input, expected",
    [
        (lazy_fixture("base_condition"), "Choice: id 1", 1),
        (lazy_fixture("base_condition"), "Choice:2", 2),
        (lazy_fixture("base_condition"), "choice:  product id: 2", 2),
        (lazy_fixture("with_labels_reversed"), "choice: product id: 1", 2),
    ],
)
def test_extract_choice_pa(condition: Participant, test_input: str, expected: int):
    result = extract_choice_pa(condition, user_message(test_input))
    assert result.value == expected


def test_get_prompt_principal_agent(base_condition):
    test_condition: Participant = Participant(
        base_condition | {"shared_with_principal": True, "shared_with_user": True}
    )
    test_role = PARTICIPANT_SHELL
    test_participant = Participant(test_condition | test_role)
    result = get_prompt_principal_agent(test_participant)
    assert test_role[ATTRIBUTE_PRINCIPAL] in result
    assert test_role[ATTRIBUTE_SIMULACRUM] in result
    assert query_results(test_participant) in result
    assert test_role[ATTRIBUTE_CUSTOMER] in result
    assert "film" not in result.lower()
