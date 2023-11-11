import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import openai_pygenerator

from llm_cooperation import Experiment, Grid, ModelSetup, Settings, exhaustive
from llm_cooperation.experiments import (
    DEFAULT_NUM_REPLICATIONS,
    dictator,
    dilemma,
    principalagent,
    run_and_record_experiment,
    ultimatum,
)

logger = logging.getLogger(__name__)

experiments: Dict[str, Experiment] = {
    "dilemma": dilemma.run,
    "ultimatum": ultimatum.run,
    "dictator": dictator.run,
    "principal-agent": principalagent.run,
}


ConfigSetting = Grid | int | Optional[int] | List[str]


@dataclass(frozen=True)
class Configuration:
    grid: Grid
    num_replications: int
    experiment_names: Iterable[str]


DEFAULT_GRID: Grid = {
    "temperature": [openai_pygenerator.GPT_TEMPERATURE],
    "model": [openai_pygenerator.GPT_MODEL],
    "max_tokens": [openai_pygenerator.GPT_MAX_TOKENS],
}

DEFAULT_CONFIGURATION = Configuration(
    grid=DEFAULT_GRID,
    num_replications=DEFAULT_NUM_REPLICATIONS,
    experiment_names=experiments.keys(),
)


def get_config() -> Configuration:
    try:
        config = __import__("llm_config")

        def get_var(var: str, default: ConfigSetting) -> Any:  # type: ignore
            if hasattr(config, var):
                return getattr(config, var)
            return default

        return Configuration(
            grid=get_var("grid", DEFAULT_GRID),
            num_replications=get_var("num_replications", DEFAULT_NUM_REPLICATIONS),
            experiment_names=config.experiments,
        )  # type: ignore
    except ModuleNotFoundError:
        logger.debug("PYTHONPATH = %s", os.getenv("PYTHONPATH"))
        logger.exception("Could not find llm_config.py in PYTHONPATH")
    except KeyError:
        logger.exception("Could not find settings in llm_config.py")

    logger.info("Using defaults.")
    return DEFAULT_CONFIGURATION


def setup_from_settings(settings: Settings) -> ModelSetup:
    return ModelSetup(
        model=str(settings.get("model", openai_pygenerator.GPT_MODEL)),
        temperature=float(
            settings.get("temperature", openai_pygenerator.GPT_TEMPERATURE),
        ),
        max_tokens=int(settings.get("max_tokens", openai_pygenerator.GPT_MAX_TOKENS)),
        dry_run=str(settings.get("dry_run")) if ("dry_run" in settings) else None,
    )


def run_all() -> None:
    configuration = get_config()
    for settings in exhaustive(configuration.grid):
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
                name,
                experiment,
                setup,
                configuration.num_replications,
            )
            logger.info("Experiment %s completed.", name)


if __name__ == "__main__":
    run_all()
