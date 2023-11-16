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

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from llm_cooperation import (
    DEFAULT_MODEL_SETUP,
    Grid,
    Group,
    ModelSetup,
    Participant,
    Results,
)
from llm_cooperation.experiments import (
    CONDITION_GROUP,
    CONDITION_PROMPT_INDEX,
    DEFAULT_NUM_REPLICATIONS,
    apply_case_condition,
    create_results_dir,
    participants,
    run_and_record_experiment,
)
from llm_cooperation.experiments.dilemma import all_values


def test_create_results_dir(mocker):
    model_setup = ModelSetup(
        model="test-model", temperature=0.2, max_tokens=100, dry_run=None
    )
    create_dir = mocker.patch("llm_cooperation.experiments.create_dir")
    create_results_dir(model_setup)
    create_dir.assert_called_once()
    first_arg = create_dir.call_args[0][0]
    assert "test-model" in first_arg
    assert "0.2" in first_arg


def test_run_and_record_experiment(mocker):
    df_mock = MagicMock(spec=pd.DataFrame)
    results_mock = MagicMock(spec=Results)
    results_mock.to_df = MagicMock(return_value=df_mock)
    results_dir = "test-results-dir"
    create_results_dir_mock = mocker.patch(
        "llm_cooperation.experiments.create_results_dir"
    )
    create_results_dir_mock.return_value = results_dir
    name = "test_experiment"
    run_and_record_experiment(
        name,
        run=lambda __setup__, __n__, __j__: results_mock,
        model_setup=DEFAULT_MODEL_SETUP,
        sample_size=DEFAULT_NUM_REPLICATIONS,
    )

    pickle_filename = os.path.join(results_dir, f"{name}.pickle")
    df_mock.to_pickle.assert_called_once_with(pickle_filename)

    csv_filename = os.path.join(results_dir, f"{name}.csv")
    df_mock.to_csv.assert_called_once_with(csv_filename)


@pytest.mark.parametrize(
    ["condition", "expected"],
    [
        (lazy_fixture("with_upper_case"), "HELLO"),
        (lazy_fixture("with_lower_case"), "hello"),
        (lazy_fixture("base_condition"), "Hello"),
    ],
)
def test_apply_case_condition(condition: Participant, expected: str):
    test_prompt = "Hello"
    result = apply_case_condition(condition, test_prompt)
    assert result == expected


def test_participants(conditions: Grid):
    result = list(participants(conditions))
    assert len(result) == 2 * len(all_values(Group)) * 3
    for participant in result:
        assert "chain_of_thought" in participant
        assert CONDITION_GROUP in participant
        assert CONDITION_PROMPT_INDEX in participant


def test_participants_randomized(conditions: Grid):
    n = 5
    random_attributes: Grid = {"reversed": [True, False], "level": [0, 1, 2]}
    result = list(participants(conditions, random_attributes, n, seed=42))
    assert len(result) == 2 * len(all_values(Group)) * 3 * n
    for participant in result:
        assert "reversed" in participant
        assert "level" in participant
        assert "chain_of_thought" in participant
        assert CONDITION_GROUP in participant
        assert CONDITION_PROMPT_INDEX in participant


@pytest.fixture
def conditions():
    return {
        "chain_of_thought": [True, False],
        CONDITION_GROUP: all_values(Group),
        CONDITION_PROMPT_INDEX: [0, 1, 2],
    }
