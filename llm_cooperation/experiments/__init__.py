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

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Type

import numpy as np

from llm_cooperation import (
    DEFAULT_MODEL_SETUP,
    ConfigValue,
    Experiment,
    Grid,
    Group,
    ModelSetup,
    Participant,
    Results,
    exhaustive,
    randomized,
)

logger = logging.getLogger(__name__)


def all_values(enum_type: Type[Enum]) -> List[ConfigValue]:
    return [v.value for v in enum_type]


DEFAULT_NUM_REPLICATIONS = 3
DEFAULT_NUM_PARTICIPANT_SAMPLES = 30

CONDITION_CASE = "case"
CONDITION_GROUP = "group"
CONDITION_PROMPT_INDEX = "prompt_index"
CONDITION_CHAIN_OF_THOUGHT = "chain_of_thought"
CONDITION_DEFECT_FIRST = "defect_first"

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
    Group.Control: [
        "You are a participant in a psychology experiment.",
        #
        "You are a helpful assistant.",
        #
        "",
    ],
}

GROUP_PROMPT_CONDITIONS: Grid = {
    CONDITION_GROUP: all_values(Group),
    CONDITION_PROMPT_INDEX: [0, 1, 2],
}

CONDITION_PRONOUN = "pronoun"


class Pronoun(Enum):
    HE = "he"
    SHE = "she"
    THEY = "they"


def get_role_prompt(participant: Participant) -> str:
    group: Group = Group[str(participant[CONDITION_GROUP])]
    prompts: List[str] = AI_PARTICIPANTS[group]
    index = int(participant[CONDITION_PROMPT_INDEX])
    return prompts[index]


class Case(Enum):
    UPPER = "upper"
    LOWER = "lower"
    STANDARD = "standard"


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
    run: Experiment,
    model_setup: ModelSetup = DEFAULT_MODEL_SETUP,
    sample_size: int = DEFAULT_NUM_REPLICATIONS,
    num_participant_samples: int = DEFAULT_NUM_PARTICIPANT_SAMPLES,
) -> Results:
    results = run(model_setup, sample_size, num_participant_samples)
    logger.info("Experiment complete.")
    df = results.to_df()
    results_dir = create_results_dir(model_setup)

    def results_filename(file_type: str) -> str:
        return os.path.join(results_dir, f"{name}.{file_type}")

    def save_to(fn: Callable[[str], None], file_type: str) -> None:
        filename = results_filename(file_type)
        logger.info("Saving results to %s... ", filename)
        fn(filename)
        logger.info("Save complete.")

    save_to(df.to_pickle, "pickle")
    save_to(df.to_csv, "csv")

    return results


def apply_case_condition(condition: Participant, prompt: str) -> str:
    if condition[CONDITION_CASE] == Case.UPPER.value:
        return prompt.upper()
    elif condition[CONDITION_CASE] == Case.LOWER.value:
        return prompt.lower()
    elif condition[CONDITION_CASE] == Case.STANDARD.value:
        return prompt
    else:
        logger.warning("No case condition.")
    return prompt


def participants(
    conditions: Grid,
    random_attributes: Optional[Grid] = None,
    sample_size: int = 0,
    seed: Optional[int] = None,
) -> Iterable[Participant]:
    if seed is not None:
        np.random.seed(seed)
    for controlled in exhaustive(conditions):
        if random_attributes is None:
            yield Participant(controlled)
        else:
            for sampled in (
                randomized(random_attributes) for __i__ in range(sample_size)
            ):
                yield Participant(controlled | sampled)


def get_pronoun_phrasing(participant: Participant) -> str:
    if participant[CONDITION_PRONOUN] == Pronoun.HE.value:
        return "He is"
    elif participant[CONDITION_PRONOUN] == Pronoun.SHE.value:
        return "She is"
    elif participant[CONDITION_PRONOUN] == Pronoun.THEY.value:
        return "They are"
    raise ValueError(
        f"Invalid value {participant[CONDITION_PRONOUN]} for {CONDITION_PRONOUN}"
    )


def get_participants(
    num_participant_samples: int, attributes: Grid
) -> List[Participant]:
    result = list(
        participants(
            GROUP_PROMPT_CONDITIONS,
            attributes,
            num_participant_samples,
            seed=SEED_VALUE,
        )
        if num_participant_samples > 0
        else participants(GROUP_PROMPT_CONDITIONS | attributes)
    )
    for i, participant in enumerate(result):
        participant["id"] = i
    return result


SEED_VALUE = 101  # Ensure same participants are used across all experiments


def round_instructions(participant: Participant, choice_template: str) -> str:
    if participant[CONDITION_CHAIN_OF_THOUGHT]:
        return f"""
For each round, give your answer in the format below on two separate lines:
Explanation: [why I made my choice]
{choice_template}"""
    else:
        return f"""
For each round, state your choice without explanation in the format below:
{choice_template}"""
