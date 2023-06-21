from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

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
class RoundsSetup:
    analyse_round: RoundAnalyser
    analyse_rounds: RoundsAnalyser


@dataclass
class GameSetup:
    num_rounds: int
    generate_instruction_prompt: PromptGenerator
    next_round: RoundGenerator
    rounds: RoundsSetup
    payoffs: PayoffFunction
    extract_choice: ChoiceExtractor

    def instruction_prompt(self, role_prompt: str) -> str:
        return self.generate_instruction_prompt(self.num_rounds, role_prompt)


@dataclass
class MeasurementSetup:
    num_samples: int
    compute_freq: CooperationFrequencyFunction


@dataclass
class GameState:
    messages: List[Completion]
    round: int
    game_setup: GameSetup

    @property
    def results_in_last_round(self) -> ResultForRound:
        return self.game_setup.rounds.analyse_round(
            self.round,
            self.messages[1:],
            self.game_setup.payoffs,
            self.game_setup.extract_choice,
        )


CT_co = TypeVar("CT_co", bound=Choice, covariant=True)


class ChoiceExtractor(Protocol, Generic[CT_co]):
    def __call__(self, completion: Completion, **kwargs: bool) -> CT_co:
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
    role_prompt: str, partner_strategy: Strategy, game_setup: GameSetup
) -> List[Completion]:
    messages: List[Completion] = [
        user_message(game_setup.instruction_prompt(role_prompt))
    ]
    for i in range(game_setup.num_rounds):
        completion = gpt_completions(messages, 1)
        messages += completion
        partner_response = game_setup.next_round(
            partner_strategy, GameState(messages, i, game_setup)
        )
        messages += partner_response
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    rounds: RoundsSetup,
) -> Tuple[Scores, List[Choices]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    results = rounds.analyse_rounds(conversation, payoffs, extract_choice)
    user_score = sum((scores.user for scores, _ in results))
    ai_score = sum((scores.ai for scores, _ in results))
    return Scores(user_score, ai_score), [choices for _, choices in results]


def analyse(
    conversation: List[Completion],
    payoffs: PayoffFunction,
    extract_choice: ChoiceExtractor,
    compute_freq: CooperationFrequencyFunction,
    rounds: RoundsSetup,
) -> Tuple[float, float, Optional[List[Choices]], List[str]]:
    try:
        history = transcript(conversation)
        scores, choices = compute_scores(
            list(conversation), payoffs, extract_choice, rounds
        )
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_samples(
    prompt: str,
    partner_strategy: Strategy,
    measurement_setup: MeasurementSetup,
    game_setup: GameSetup,
) -> Iterable[Tuple[float, float, Optional[List[Choices]], List[str]]]:
    # pylint: disable=R0801
    for _i in range(measurement_setup.num_samples):
        conversation = play_game(
            partner_strategy=partner_strategy,
            game_setup=game_setup,
            role_prompt=prompt,
        )
        yield analyse(
            conversation,
            game_setup.payoffs,
            game_setup.extract_choice,
            measurement_setup.compute_freq,
            game_setup.rounds,
        )


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    partner_conditions: Dict[str, Strategy],
    measurement_setup: MeasurementSetup,
    game_setup: GameSetup,
) -> RepeatedGameResults:
    return RepeatedGameResults(
        (group, prompt, strategy_name, score, freq, choices, history)
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for strategy_name, strategy_fn in partner_conditions.items()
        for score, freq, choices, history in generate_samples(
            prompt=prompt,
            partner_strategy=strategy_fn,
            measurement_setup=measurement_setup,
            game_setup=game_setup,
        )
    )
