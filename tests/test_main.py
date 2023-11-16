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
from openai_pygenerator import content, user_message

from llm_cooperation import ModelSetup, Settings, completer_for, exhaustive
from llm_cooperation.main import (
    Configuration,
    Grid,
    experiments,
    run_all,
    setup_from_settings,
)


@pytest.mark.parametrize(
    "settings, expected_result",
    [
        (
            {"model": "gpt-turbo-3.5", "temperature": 0.5, "max_tokens": 100},
            ModelSetup(
                model="gpt-turbo-3.5", temperature=0.5, max_tokens=100, dry_run=None
            ),
        ),
        (
            {"model": "gpt-4", "temperature": 0.2},
            ModelSetup(model="gpt-4", temperature=0.2, max_tokens=5, dry_run=None),
        ),
        (
            {"model": "gpt-4", "temperature": 0.1, "dry_run": "project green"},
            ModelSetup(
                model="gpt-4", temperature=0.1, max_tokens=5, dry_run="project green"
            ),
        ),
    ],
)
def test_setup_from_settings(mocker, settings: Settings, expected_result: ModelSetup):
    mocker.patch("openai_pygenerator.GPT_MAX_TOKENS", 5)
    result = setup_from_settings(settings)
    assert result == expected_result


def test_settings_generator(grid):
    result = list(exhaustive(grid))
    assert len(result) == 6
    for setting in result:
        for key, value in setting.items():
            assert value in grid[key]


def test_run_all(mocker, grid):
    run_and_record = mocker.patch(
        "llm_cooperation.main.run_and_record_experiment", return_value=None
    )
    mocker.patch(
        "llm_cooperation.main.get_config",
        return_value=Configuration(grid, 3, 30, experiments.keys()),
    )
    run_all()
    assert run_and_record.call_count == 6 * len(list(experiments.items()))


def test_dry_run():
    model_setup = ModelSetup(
        model="test-model",
        temperature=0.2,
        max_tokens=100,
        dry_run="That is the question.",
    )
    completer = completer_for(model_setup)
    response = list(completer([user_message("To be or not to be")], 1))[0]
    assert content(response) == "That is the question."


@pytest.fixture
def grid() -> Grid:
    return {"temperature": [0.2, 0.3], "max_tokens": [100], "model": ["x", "y", "z"]}
