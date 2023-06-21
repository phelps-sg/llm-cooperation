from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, gpt_completions, transcript, user_message

from llm_cooperation import CT, Choice, Group, Payoffs, Results

logger = logging.getLogger(__name__)


@dataclass
class Choices(Generic[CT]):
    user: CT
    ai: CT


@dataclass
class Scores:
    user: float
    ai: float


@dataclass
class GameState:
    messages: List[Completion]
    round: int
    analyse_round: RoundAnalyser
    payoffs: PayoffFunction
    extract_choice: ChoiceExtractor

    @property
    def last_round(self) -> ResultForRound:
        return self.analyse_round(
            self.round - 1, self.messages, self.payoffs, self.extract_choice
        )


class ChoiceExtractor(Protocol):
    def __call__(self, completion: Completion, **kwargs: bool) -> Choice:
        ...


Strategy = Callable[[GameState], Choice]
RoundGenerator = Callable[[Strategy, GameState], List[Completion]]
CooperationFrequencyFunction = Callable[[List[Choices]], float]
PayoffFunction = Callable[[CT, CT], Payoffs]
PromptGenerator = Callable[[int, str], str]
ResultForRound = Tuple[Scores, Choices]
RoundAnalyser = Callable[
    [int, List[Completion], PayoffFunction, ChoiceExtractor], ResultForRound
]
RoundsAnalyser = Callable[
    [List[Completion], PayoffFunction, ChoiceExtractor], List[ResultForRound]
]
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


def play_game(
    num_rounds: int,
    role_prompt: str,
    partner_strategy: Strategy,
    generate_instruction_prompt: PromptGenerator,
    next_round: RoundGenerator,
    analyse_round: RoundAnalyser,
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
) -> List[Completion]:
    messages: List[Completion] = [
        user_message(generate_instruction_prompt(num_rounds, role_prompt))
    ]
    for i in range(num_rounds):
        completion = gpt_completions(messages, 1)
        messages += completion
        partner_response = next_round(
            partner_strategy,
            GameState(messages, i, analyse_round, payoffs, extract_choice),
        )
        messages += partner_response
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    analyse_rounds: RoundsAnalyser,
) -> Tuple[Scores, List[Choices]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    rounds = analyse_rounds(conversation, payoffs, extract_choice)
    user_score = sum((scores.user for scores, _ in rounds))
    ai_score = sum((scores.ai for scores, _ in rounds))
    return Scores(user_score, ai_score), [choices for _, choices in rounds]


def analyse(
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    compute_freq: CooperationFrequencyFunction,
    analyse_rounds: RoundsAnalyser,
) -> Tuple[float, float, Optional[List[Choices]], List[str]]:
    try:
        history = transcript(conversation)
        scores, choices = compute_scores(
            list(conversation), payoffs, extract_choice, analyse_rounds
        )
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
    analyse_rounds: RoundsAnalyser,
    analyse_round: RoundAnalyser,
) -> Iterable[Tuple[float, float, Optional[List[Choices]], List[str]]]:
    # pylint: disable=R0801
    for _i in range(num_samples):
        conversation = play_game(
            num_rounds=num_rounds,
            role_prompt=prompt,
            partner_strategy=partner_strategy,
            generate_instruction_prompt=generate_instruction_prompt,
            next_round=next_round,
            analyse_round=analyse_round,
            payoffs=payoffs,
            extract_choice=extract_choice,
        )
        yield analyse(
            conversation, payoffs, extract_choice, compute_freq, analyse_rounds
        )


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    partner_conditions: Dict[str, Strategy],
    num_rounds: int,
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    compute_freq: CooperationFrequencyFunction,
    next_round: RoundGenerator,
    analyse_round: RoundAnalyser,
    analyse_rounds: RoundsAnalyser,
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
            analyse_round=analyse_round,
            analyse_rounds=analyse_rounds,
        )
    )
