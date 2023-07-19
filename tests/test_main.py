import pytest

from llm_cooperation import ModelSetup
from llm_cooperation.main import (
    Configuration,
    Grid,
    Settings,
    experiments,
    run_all,
    settings_generator,
    setup_from_settings,
)


@pytest.mark.parametrize(
    "settings, expected_result",
    [
        (
            {"model": "gpt-turbo-3.5", "temperature": 0.5, "max_tokens": 100},
            ModelSetup(model="gpt-turbo-3.5", temperature=0.5, max_tokens=100),
        ),
        (
            {"model": "gpt-4", "temperature": 0.2},
            ModelSetup(model="gpt-4", temperature=0.2, max_tokens=5),
        ),
    ],
)
def test_setup_from_settings(mocker, settings: Settings, expected_result: ModelSetup):
    mocker.patch("openai_pygenerator.GPT_MAX_TOKENS", 5)
    result = setup_from_settings(settings)
    assert result == expected_result


def test_settings_generator(grid):
    result = list(settings_generator(grid))
    for setting in result:
        for key, value in setting.items():
            assert value in grid[key]


def test_run_all(mocker, grid):
    run_and_record = mocker.patch(
        "llm_cooperation.main.run_and_record_experiment", return_value=None
    )
    mocker.patch(
        "llm_cooperation.main.get_config",
        return_value=Configuration(grid, 30, experiments.keys()),
    )
    run_all()
    assert run_and_record.call_count == 6 * len(list(experiments.items()))


@pytest.fixture
def grid() -> Grid:
    return {"temperature": [0.2, 0.3], "max_tokens": [100], "model": ["x", "y", "z"]}
