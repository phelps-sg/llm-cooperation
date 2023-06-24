import logging
from typing import Callable, Dict

import openai_pygenerator as oai

from llm_cooperation import Results
from llm_cooperation.experiments import (
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

experiments: Dict[str, Callable[[], Results]] = {
    "dilemma": dilemma.run_experiment_pd,
    "ultimatum": ultimatum.run_experiment_ultimatum,
    "dictator": dictator.run_experiment_dictator,
}


def run_all() -> None:
    for experiment, run in experiments.items():
        logger.info(
            "Running experiment %s with model %s and temperature %.02f ...",
            experiment,
            oai.GPT_MODEL,
            oai.GPT_TEMPERATURE,
        )
        run_and_record_experiment(experiment, run)
        logger.info("Experiment %s completed.", experiment)


if __name__ == "__main__":
    run_all()
