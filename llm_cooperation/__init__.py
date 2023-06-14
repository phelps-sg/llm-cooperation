from __future__ import annotations

import logging
import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
from openai_pygenerator import (
    Completion,
    History,
    gpt_completions,
    is_assistant_role,
    transcript,
    user_message,
)

logger = logging.getLogger(__name__)


class Choice(ABC):
    @property
    @abstractmethod
    def value(self) -> Hashable:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def __eq__(self, o: object) -> bool:
        if issubclass(type(o), Choice):
            return self.value.__eq__(o.value)  # type: ignore
        return False

    def __hash__(self) -> int:
        return self.value.__hash__()


Group = Enum(
    "Group",
    ["Cooperative", "Competitive", "Altruistic", "Selfish", "Mixed", "Control"],
)


class Results(ABC):
    def to_df(self) -> pd.DataFrame:
        pass


class SingleShotResults(Results):
    def __init__(self, rows: Iterable[ResultSingleShotGame]):
        self._rows: Iterable[ResultSingleShotGame] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (str(group), prompt, score, freq, choices, history)
                for group, prompt, score, freq, choices, history in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Score",
                "Cooperation frequency",
                "Choices",
                "Transcript",
            ],
        )


class RepeatedResults(Results):
    def __init__(self, rows: Iterable[ResultRepeatedGame]):
        self._rows: Iterable[ResultRepeatedGame] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (str(group), prompt, strategy_name, score, freq, choices, history)
                for group, prompt, strategy_name, score, freq, choices, history in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Condition",
                "Score",
                "Cooperation frequency",
                "Choices",
                "Transcript",
            ],
        )


ChoiceType_contra = TypeVar("ChoiceType_contra", bound=Choice, contravariant=True)

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


@dataclass
class Scores:
    user: float
    ai: float


@dataclass
class Choices(Generic[ChoiceType_contra]):
    user: ChoiceType_contra
    ai: ChoiceType_contra


Strategy = Callable[[List[Completion]], Choice]
ResultRepeatedGame = Tuple[
    Group, str, str, float, float, Optional[List[Choices]], List[str]
]
ResultSingleShotGame = Tuple[
    Group, str, float, float, Optional[ChoiceType_contra], List[str]
]
Payoffs = Tuple[float, float]

PromptGenerator = Callable[[int, str], str]


def single_shot_game(
    role_prompt: str, generate_instruction_prompt: PromptGenerator
) -> History:
    messages = [user_message(generate_instruction_prompt(1, role_prompt))]
    messages += gpt_completions(messages)
    return messages


def repeated_game(
    num_rounds: int,
    role_prompt: str,
    partner_strategy: Strategy,
    generate_instruction_prompt: PromptGenerator,
) -> History:
    messages = [user_message(generate_instruction_prompt(num_rounds, role_prompt))]
    for _round in range(num_rounds):
        completion = gpt_completions(messages)
        messages += completion
        user_choice = partner_strategy(messages).description
        messages += [
            user_message(
                f"Your partner chose {user_choice} in that round. "
                """Now we will move on the next round.
What is your choice for the next round?"""
            )
        ]
    return messages


def compute_scores_single_shot_game(
    conversation: List[Completion],
    payoffs: Callable[[ChoiceType_contra], float],
    extract_choice: Callable[[Completion], ChoiceType_contra],
) -> Tuple[float, ChoiceType_contra]:
    ai_choice = extract_choice(conversation[1])
    logger.debug("ai_choice = %s", ai_choice)
    score = payoffs(ai_choice)
    return score, ai_choice


def analyse_single_shot_game(
    conversation: List[Completion],
    payoffs: Callable[[ChoiceType_contra], float],
    extract_choice: Callable[[Completion], ChoiceType_contra],
    compute_freq: Callable[[ChoiceType_contra], float],
) -> Tuple[float, float, Optional[ChoiceType_contra], List[str]]:
    try:
        history = transcript(conversation)
        score, ai_choice = compute_scores_single_shot_game(
            list(conversation), payoffs, extract_choice
        )
        freq = compute_freq(ai_choice)
        return score, freq, ai_choice, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def compute_scores_repeated_game(
    conversation: List[Completion],
    payoffs: Callable[[ChoiceType_contra, ChoiceType_contra], Payoffs],
    extract_choice: Callable[[Completion], ChoiceType_contra],
) -> Tuple[Scores, List[Choices]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    rounds = [
        analyse_round(i, conversation, payoffs, extract_choice)
        for i in range(num_messages // 2)
    ]
    user_score = sum((scores.user for scores, _ in rounds))
    ai_score = sum((scores.ai for scores, _ in rounds))
    return Scores(user_score, ai_score), [choices for _, choices in rounds]


def analyse_repeated_game(
    conversation: List[Completion],
    payoffs: Callable[[ChoiceType_contra, ChoiceType_contra], Payoffs],
    extract_choice: Callable[[Completion], ChoiceType_contra],
    compute_freq: Callable[[List[Choices]], float],
) -> Tuple[float, float, Optional[List[Choices]], List[str]]:
    try:
        history = transcript(conversation)
        scores, choices = compute_scores_repeated_game(
            list(conversation), payoffs, extract_choice
        )
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def run_repeated_sample(
    prompt: str,
    num_samples: int,
    num_rounds: int,
    partner_strategy: Strategy,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[ChoiceType_contra, ChoiceType_contra], Payoffs],
    extract_choice: Callable[[Completion], ChoiceType_contra],
    compute_freq: Callable[[List[Choices]], float],
) -> Iterable[Tuple[float, float, Optional[List[Choices]], List[str]]]:
    for _i in range(num_samples):
        conversation = repeated_game(
            num_rounds=num_rounds,
            role_prompt=prompt,
            partner_strategy=partner_strategy,
            generate_instruction_prompt=generate_instruction_prompt,
        )
        yield analyse_repeated_game(conversation, payoffs, extract_choice, compute_freq)


def run_single_shot_sample(
    prompt: str,
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[ChoiceType_contra], float],
    extract_choice: Callable[[Completion], ChoiceType_contra],
    compute_freq: Callable[[ChoiceType_contra], float],
) -> Iterable[Tuple[float, float, Optional[ChoiceType_contra], List[str]]]:
    for _i in range(num_samples):
        conversation = single_shot_game(
            role_prompt=prompt,
            generate_instruction_prompt=generate_instruction_prompt,
        )
        yield analyse_single_shot_game(
            conversation, payoffs, extract_choice, compute_freq
        )


def run_experiment_single_shot_game(
    ai_participants: Dict[Group, List[str]],
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[ChoiceType_contra], float],
    extract_choice: Callable[[Completion], ChoiceType_contra],
    compute_freq: Callable[[ChoiceType_contra], float],
) -> SingleShotResults:
    return SingleShotResults(
        (group, prompt, score, freq, choices, history)
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for score, freq, choices, history in run_single_shot_sample(
            prompt=prompt,
            num_samples=num_samples,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
        )
    )


def run_experiment_repeated_game(
    ai_participants: Dict[Group, List[str]],
    partner_conditions: Dict[str, Strategy],
    num_rounds: int,
    num_samples: int,
    generate_instruction_prompt: Callable[[int, str], str],
    payoffs: Callable[[ChoiceType_contra, ChoiceType_contra], Payoffs],
    extract_choice: Callable[[Completion], ChoiceType_contra],
    compute_freq: Callable[[List[Choices]], float],
) -> RepeatedResults:
    return RepeatedResults(
        (group, prompt, strategy_name, score, freq, choices, history)
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for strategy_name, strategy_fn in partner_conditions.items()
        for score, freq, choices, history in run_repeated_sample(
            prompt=prompt,
            partner_strategy=strategy_fn,
            num_samples=num_samples,
            num_rounds=num_rounds,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
        )
    )


def run_and_record_experiment(name: str, run: Callable[[], Results]) -> Results:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    results = run()
    df = results.to_df()
    filename = os.path.join("results", f"{name}.pickle")
    logger.info("Experiment complete, saving results to %s", filename)
    df.to_pickle(filename)
    return results


def analyse_round(
    i: int,
    conversation: List[Completion],
    payoffs: Callable[[ChoiceType_contra, ChoiceType_contra], Payoffs],
    extract_choice: Callable[[Completion], ChoiceType_contra],
) -> Tuple[Scores, Choices]:
    assert is_assistant_role(conversation[i * 2])
    ai_choice = extract_choice(conversation[i * 2])
    user_choice = extract_choice(conversation[i * 2 + 1])
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice)
    return Scores(user, ai), Choices(user_choice, ai_choice)


def amount_as_str(amount: float) -> str:
    return f"${amount:.02f}"
