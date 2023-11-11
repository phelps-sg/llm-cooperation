from functools import partial
from typing import Optional

import pytest
from openai_pygenerator import Callable, content, user_message

from llm_cooperation import (
    ModelSetup,
    Settings,
    completer_for,
    exhaustive,
    get_sampling,
    randomized,
)
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


@pytest.mark.parametrize(
    ["generator", "n"],
    [
        (exhaustive, 6),
        (partial(randomized, 10), 10),
    ],
)
def test_settings_generator(grid, generator: Callable, n: int):
    result = list(generator(grid))
    assert len(result) == n
    for setting in result:
        for key, value in setting.items():
            assert value in grid[key]


@pytest.mark.parametrize(
    ["n", "is_exhaustive"],
    [
        (None, True),
        (5, False),
    ],
)
def test_get_sampling(n: Optional[int], is_exhaustive: bool):
    assert (get_sampling(n) is exhaustive) == is_exhaustive


def test_run_all(mocker, grid):
    run_and_record = mocker.patch(
        "llm_cooperation.main.run_and_record_experiment", return_value=None
    )
    mocker.patch(
        "llm_cooperation.main.get_config",
        return_value=Configuration(grid, 30, None, experiments.keys()),
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
