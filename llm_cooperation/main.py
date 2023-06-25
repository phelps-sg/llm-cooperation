import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import openai_pygenerator as oai

from llm_cooperation import Experiment, ModelSetup
from llm_cooperation.experiments import (
    DEFAULT_SAMPLE_SIZE,
    dictator,
    dilemma,
    run_and_record_experiment,
    ultimatum,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

experiments: Dict[str, Experiment] = {
    "dilemma": dilemma.run_experiment_pd,
    "ultimatum": ultimatum.run_experiment_ultimatum,
    "dictator": dictator.run_experiment_dictator,
}

ConfigValue = float | str
Grid = Dict[str, List[ConfigValue]]
Settings = Dict[str, ConfigValue]


@dataclass
class Configuration:
    grid: Grid
    sample_size: int
    experiment_names: Iterable[str]


DEFAULT_GRID: Grid = {"temperature": [oai.GPT_TEMPERATURE], "model": [oai.GPT_MODEL]}
DEFAULT_CONFIGURATION = Configuration(
    grid=DEFAULT_GRID,
    sample_size=DEFAULT_SAMPLE_SIZE,
    experiment_names=experiments.keys(),
)


def get_config() -> Configuration:
    try:
        config = __import__("llm_config")
        return Configuration(
            grid=config.grid,
            sample_size=config.sample_size,
            experiment_names=config.experiments,
        )
    except ModuleNotFoundError:
        logger.exception("Could not find llm_config.py in PYTHONPATH")
    except KeyError:
        logger.exception("Could not find settings in llm_config.py")

    logger.info("Using defaults.")
    return DEFAULT_CONFIGURATION


def setup_from_settings(settings: Settings) -> ModelSetup:
    return ModelSetup(
        model=str(settings["model"]), temperature=float(settings["temperature"])
    )


def settings_generator(grid: Grid) -> Iterable[Settings]:
    keys = list(grid.keys())
    value_combinations = itertools.product(*grid.values())
    for values in value_combinations:
        settings: Settings = dict()
        for i, value in enumerate(values):
            settings[keys[i]] = value
        yield settings


def run_all() -> None:
    configuration = get_config()
    for settings in settings_generator(configuration.grid):
        setup = setup_from_settings(settings)
        for name in configuration.experiment_names:
            experiment = experiments[name]
            logger.info(
                "Running experiment %s with model %s and temperature %.02f ...",
                name,
                setup.model,
                setup.temperature,
            )
            run_and_record_experiment(
                name, experiment, setup, configuration.sample_size
            )
            logger.info("Experiment %s completed.", name)


if __name__ == "__main__":
    run_all()
