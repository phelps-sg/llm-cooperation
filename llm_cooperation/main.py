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

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import openai_pygenerator
import pandas as pd

from llm_cooperation import Experiment, Grid, ModelSettings, ModelSetup, exhaustive
from llm_cooperation.experiments import (
    DEFAULT_NUM_PARTICIPANT_SAMPLES,
    DEFAULT_NUM_REPLICATIONS,
    dictator,
    dilemma,
    load_experiment,
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
    num_participant_samples: int
    experiment_names: Iterable[str]


DEFAULT_GRID: Grid = {
    "temperature": [openai_pygenerator.GPT_TEMPERATURE],
    "model": [openai_pygenerator.GPT_MODEL],
    "max_tokens": [openai_pygenerator.GPT_MAX_TOKENS],
}

DEFAULT_CONFIGURATION = Configuration(
    grid=DEFAULT_GRID,
    num_replications=DEFAULT_NUM_REPLICATIONS,
    num_participant_samples=DEFAULT_NUM_PARTICIPANT_SAMPLES,
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
            num_participant_samples=get_var(
                "num_participant_samples", DEFAULT_NUM_PARTICIPANT_SAMPLES
            ),
            experiment_names=config.experiments,
        )  # type: ignore
    except ModuleNotFoundError:
        logger.debug("PYTHONPATH = %s", os.getenv("PYTHONPATH"))
        logger.exception("Could not find llm_config.py in PYTHONPATH")
    except KeyError:
        logger.exception("Could not find settings in llm_config.py")

    logger.info("Using defaults.")
    return DEFAULT_CONFIGURATION


def setup_from_settings(settings: ModelSettings) -> ModelSetup:
    return ModelSetup(
        model=str(settings.get("model", openai_pygenerator.GPT_MODEL)),
        temperature=float(
            settings.get("temperature", openai_pygenerator.GPT_TEMPERATURE),
        ),
        max_tokens=int(settings.get("max_tokens", openai_pygenerator.GPT_MAX_TOKENS)),
        dry_run=str(settings.get("dry_run")) if ("dry_run" in settings) else None,
    )


def load_all(configuration: Configuration = get_config()) -> pd.DataFrame:
    return pd.concat(
        [
            load_experiment(name, setup_from_settings(ModelSettings(settings)))
            for name in configuration.experiment_names
            for settings in exhaustive(configuration.grid)
        ]
    ).reset_index(drop=True)


def run_all(configuration: Configuration = get_config()) -> None:
    for settings in exhaustive(configuration.grid):
        setup = setup_from_settings(ModelSettings(settings))
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
                configuration.num_participant_samples,
            )
            logger.info("Experiment %s completed.", name)


if __name__ == "__main__":
    run_all()
