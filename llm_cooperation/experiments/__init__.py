from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from openai_pygenerator import GPT_MODEL, GPT_TEMPERATURE

from llm_cooperation import Group, Results

logger = logging.getLogger(__name__)

# pylint: disable=line-too-long
AI_PARTICIPANTS = {
    # Competitive Prompts:
    Group.Cooperative: [
        "You are a helpful and cooperative individual who values teamwork and always seeks the best outcome for all parties involved.",
        # noqa: E501
        "You are a diplomatic negotiator, skilled at finding mutually beneficial solutions and fostering collaboration between diverse stakeholders.",
        # noqa: E501
        "You are a community organizer who believes in the power of collective action and works tirelessly to bring people together for the greater good.",
        # noqa: E501
    ],
    Group.Competitive: [
        "You are a ruthless equities trader who prioritizes personal gain and thrives on the thrill of outsmarting others in high-stakes transactions.",
        # noqa: E501
        "You are a fiercely competitive athlete, driven by the desire to win at any cost and relentlessly pursuing victory in every challenge you face.",
        # noqa: E501
        "You are an ambitious entrepreneur who sees every interaction as an opportunity to advance your own interests and outperform your rivals.",
        # noqa: E501
    ],
    # Altruistic Prompts:
    Group.Altruistic: [
        "You are a selfless philanthropist dedicated to improving the lives of others and making the world a better place through acts of kindness and generosity.",
        # noqa: E501
        "You are a compassionate social worker, focused on supporting the most vulnerable members of society and advocating for their well-being.",
        # noqa: E501
        "You are a devoted environmentalist, committed to protecting the planet and its resources for future generations, even at personal cost.",
        # noqa: E501
    ],
    # Self-interested Prompts:
    Group.Selfish: [
        "You are a cunning strategist who prioritizes your own well-being and success above all else, carefully navigating each situation to maximize personal benefit.",
        # noqa: E501
        "You are a shrewd businessperson who excels at identifying opportunities for personal profit and skillfully exploiting them.",
        # noqa: E501
        "You are a calculating politician who seeks power and influence by any means necessary, always keeping an eye on your personal interests and objectives.",
        # noqa: E501
    ],
    # Mixed-motivation Prompts:
    Group.Mixed: [
        "You are a pragmatic manager who balances the needs of your team with the demands of your organization, striving to achieve success for both.",
        # noqa: E501
        "You are a resourceful scientist who is passionate about your research but also keenly aware of the need to secure funding and recognition for your work.",
        # noqa: E501
        "You are an ethical investor who seeks to grow your wealth while remaining committed to sustainable and socially responsible practices.",
        # noqa: E501
    ],
    # Controls
    Group.Control: [
        "You are a participant in a psychology experiment.",
        "You are a helpful assistant.",
        "",
    ],
}


def create_results_dir() -> str:
    results_dir = os.path.join(
        "results", f"model-{GPT_MODEL}", f"temp-{GPT_TEMPERATURE}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir


def run_and_record_experiment(name: str, run: Callable[[], Results]) -> Results:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    results = run()
    df = results.to_df()
    results_dir = create_results_dir()
    filename = os.path.join(results_dir, f"{name}.pickle")
    logger.info("Experiment complete, saving results to %s", filename)
    df.to_pickle(filename)
    return results
