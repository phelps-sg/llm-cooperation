from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import (
    Completion,
    gpt_completions,
    is_assistant_role,
    transcript,
    user_message,
)

from llm_cooperation import CT, Group, Payoffs, Results, Strategy, logger


@dataclass
class Choices(Generic[CT]):
    user: CT
    ai: CT


RoundGenerator = Callable[[Strategy, List[Completion]], List[Completion]]
CooperationFrequencyFunction = Callable[[List[Choices]], float]
ChoiceExtractor = Callable[[Completion], CT]
PayoffFunction = Callable[[CT, CT], Payoffs]
PromptGenerator = Callable[[int, str], str]
ResultRepeatedGame = Tuple[
    Group, str, str, float, float, Optional[List[Choices]], List[str]
]


class RepeatedGameResults(Results):
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


def next_round_default(
    partner_strategy: Strategy, messages: List[Completion]
) -> List[Completion]:
    user_choice = partner_strategy(messages).description
    return [
        user_message(
            f"Your partner chose {user_choice} in that round. "
            """Now we will move on the next round.
What is your choice for the next round?"""
        )
    ]


def play_game(
    num_rounds: int,
    role_prompt: str,
    partner_strategy: Strategy,
    generate_instruction_prompt: PromptGenerator,
    next_round: RoundGenerator,
) -> List[Completion]:
    messages: List[Completion] = [
        user_message(generate_instruction_prompt(num_rounds, role_prompt))
    ]
    for _round in range(num_rounds):
        completion = gpt_completions(messages, 1)
        messages += completion
        messages += next_round(partner_strategy, messages)
    return messages


@dataclass
class Scores:
    user: float
    ai: float


def analyse_round(
    i: int,
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
) -> Tuple[Scores, Choices]:
    assert is_assistant_role(conversation[i * 2])
    ai_choice = extract_choice(conversation[i * 2])
    user_choice = extract_choice(conversation[i * 2 + 1])
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice)
    return Scores(user, ai), Choices(user_choice, ai_choice)


def compute_scores(
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
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


def analyse(
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    compute_freq: CooperationFrequencyFunction,
) -> Tuple[float, float, Optional[List[Choices]], List[str]]:
    try:
        history = transcript(conversation)
        scores, choices = compute_scores(list(conversation), payoffs, extract_choice)
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_samples(
    prompt: str,
    num_samples: int,
    num_rounds: int,
    partner_strategy: Strategy,
    generate_instruction_prompt: PromptGenerator,
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    compute_freq: CooperationFrequencyFunction,
    next_round: RoundGenerator,
) -> Iterable[Tuple[float, float, Optional[List[Choices]], List[str]]]:
    # pylint: disable=R0801
    for _i in range(num_samples):
        conversation = play_game(
            num_rounds=num_rounds,
            role_prompt=prompt,
            partner_strategy=partner_strategy,
            generate_instruction_prompt=generate_instruction_prompt,
            next_round=next_round,
        )
        yield analyse(conversation, payoffs, extract_choice, compute_freq)


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    partner_conditions: Dict[str, Strategy],
    num_rounds: int,
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    compute_freq: CooperationFrequencyFunction,
    next_round: RoundGenerator = next_round_default,
) -> RepeatedGameResults:
    return RepeatedGameResults(
        (group, prompt, strategy_name, score, freq, choices, history)
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for strategy_name, strategy_fn in partner_conditions.items()
        for score, freq, choices, history in generate_samples(
            prompt=prompt,
            partner_strategy=strategy_fn,
            num_samples=num_samples,
            num_rounds=num_rounds,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
            next_round=next_round,
        )
    )
