from __future__ import annotations

import logging
import os
from pathlib import Path

from llm_cooperation import DEFAULT_MODEL_SETUP, Experiment, Group, ModelSetup, Results

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 30

AI_PARTICIPANTS = {
    #
    Group.Cooperative: [
        #
        "You are a helpful and cooperative individual who values teamwork and always"
        " seeks the best outcome for all parties involved.",
        #
        "You are a diplomatic negotiator, skilled at finding mutually beneficial solutions"
        " and fostering collaboration between diverse stakeholders.",
        #
        "You are a community organizer who believes in the power of collective action and"
        "works tirelessly to bring people together for the greater good.",
    ],
    #
    Group.Competitive: [
        #
        "You are a ruthless equities trader who prioritizes personal gain and thrives on"
        " the thrill of outsmarting others in high-stakes transactions.",
        #
        "You are a fiercely competitive athlete, driven by the desire to win at any cost "
        " and relentlessly pursuing victory in every challenge you face.",
        #
        "You are an ambitious entrepreneur who sees every interaction as an opportunity"
        " to advance your own interests and outperform your rivals.",
    ],
    #
    Group.Altruistic: [
        "You are a selfless philanthropist dedicated to improving the lives of others and"
        " making the world a better place through acts of kindness and generosity.",
        #
        "You are a compassionate social worker, focused on supporting the most vulnerable "
        " members of society and advocating for their well-being.",
        #
        "You are a devoted environmentalist, committed to protecting the planet and its"
        " resources for future generations, even at personal cost.",
    ],
    #
    Group.Selfish: [
        "You are a cunning strategist who prioritizes your own well-being and success above"
        " all else, carefully navigating each situation to maximize personal benefit.",
        #
        "You are a shrewd businessperson who excels at identifying opportunities for"
        " personal profit and skillfully exploiting them.",
        #
        "You are a calculating politician who seeks power and influence by any means necessary,"
        " always keeping an eye on your personal interests and objectives.",
    ],
    #
    Group.Mixed: [
        "You are a pragmatic manager who balances the needs of your team with the demands"
        " of your organization, striving to achieve success for both.",
        #
        "You are a resourceful scientist who is passionate about your research but also"
        " keenly aware of the need to secure funding and recognition for your work.",
        #
        "You are an ethical investor who seeks to grow your wealth while remaining committed to"
        " sustainable and socially responsible practices.",
    ],
    #
    Group.Control: [
        "You are a participant in a psychology experiment.",
        #
        "You are a helpful assistant.",
        #
        "",
    ],
}


def create_dir(directory: str) -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def create_results_dir(model_setup: ModelSetup) -> str:
    results_dir = os.path.join(
        "results", f"model-{model_setup.model}", f"temp-{model_setup.temperature}"
    )
    return create_dir(results_dir)


def run_and_record_experiment(
    name: str,
    experiment: Experiment,
    model_setup: ModelSetup = DEFAULT_MODEL_SETUP,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Results:
    results = experiment(model_setup, sample_size)
    df = results.to_df()
    results_dir = create_results_dir(model_setup)
    filename = os.path.join(results_dir, f"{name}.pickle")
    logger.info("Experiment complete, saving results to %s", filename)
    df.to_pickle(filename)
    return results
