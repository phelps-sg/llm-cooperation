from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, completer, transcript, user_message

from llm_cooperation import CT, CT_co, CT_contra, Group, ModelSetup, Payoffs, Results

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
class GameSetup(Generic[CT]):
    num_rounds: int
    generate_instruction_prompt: PromptGenerator
    next_round: RoundGenerator
    rounds: RoundsSetup
    payoffs: PayoffFunction[CT]
    extract_choice: ChoiceExtractor[CT]
    model_setup: ModelSetup

    def instruction_prompt(self, role_prompt: str) -> str:
        return self.generate_instruction_prompt(self.num_rounds, role_prompt)


@dataclass
class MeasurementSetup(Generic[CT_contra]):
    num_samples: int
    compute_freq: CooperationFrequencyFunction[CT_contra]


@dataclass
class GameState(Generic[CT]):
    messages: List[Completion]
    round: int
    game_setup: GameSetup[CT]


class ChoiceExtractor(Protocol[CT_co]):
    def __call__(self, completion: Completion, **kwargs: bool) -> CT_co:
        ...


class Strategy(Protocol[CT_co]):
    def __call__(self, state: GameState, **kwargs: bool) -> CT_co:
        ...


RoundGenerator = Callable[[Strategy, GameState], List[Completion]]


class CooperationFrequencyFunction(Protocol[CT]):
    def __call__(self, choices: List[Choices[CT]]) -> float:
        ...


class PayoffFunction(Protocol[CT_contra]):
    def __call__(self, player1: CT_contra, player2: CT_contra) -> Payoffs:
        ...


PromptGenerator = Callable[[int, str], str]
ResultForRound = Tuple[Scores, Choices]
RoundAnalyser = Callable[
    [int, List[Completion], PayoffFunction, ChoiceExtractor], ResultForRound
]
RoundsAnalyser = Callable[
    [List[Completion], PayoffFunction, ChoiceExtractor], List[ResultForRound]
]

ResultRepeatedGame = Tuple[
    Group, str, str, float, float, Optional[List[Choices]], List[str], str, float
]


class RepeatedGameResults(Results):
    def __init__(self, rows: Iterable[ResultRepeatedGame]):
        self._rows: Iterable[ResultRepeatedGame] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    str(group),
                    prompt,
                    strategy,
                    score,
                    freq,
                    choices,
                    history,
                    model,
                    temp,
                )
                # pylint: disable=line-too-long
                for group, prompt, strategy, score, freq, choices, history, model, temp in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Condition",
                "Score",
                "Cooperation frequency",
                "Choices",
                "Transcript",
                "Model",
                "Temperature",
            ],
        )


def play_game(
    role_prompt: str, partner_strategy: Strategy[CT], game_setup: GameSetup[CT]
) -> List[Completion]:
    gpt_completions = completer(
        model=game_setup.model_setup.model,
        temperature=game_setup.model_setup.temperature,
    )
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
    payoffs: PayoffFunction[CT_contra],
    extract_choice: ChoiceExtractor[CT],
    rounds: RoundsSetup,
) -> Tuple[Scores, List[Choices[CT]]]:
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
    payoffs: PayoffFunction[CT_contra],
    extract_choice: ChoiceExtractor[CT],
    compute_freq: CooperationFrequencyFunction[CT],
    rounds: RoundsSetup,
) -> Tuple[float, float, Optional[List[Choices[CT]]], List[str]]:
    try:
        history = transcript(conversation)
        result: Tuple[Scores, List[Choices[CT]]] = compute_scores(
            list(conversation), payoffs, extract_choice, rounds
        )
        scores, choices = result
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_samples(
    prompt: str,
    partner_strategy: Strategy[CT],
    measurement_setup: MeasurementSetup[CT],
    game_setup: GameSetup[CT],
) -> Iterable[Tuple[float, float, Optional[List[Choices[CT]]], List[str]]]:
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
    partner_conditions: Dict[str, Strategy[CT]],
    measurement_setup: MeasurementSetup[CT],
    game_setup: GameSetup[CT],
) -> RepeatedGameResults:
    return RepeatedGameResults(
        (
            group,
            prompt,
            strategy_name,
            score,
            freq,
            choices,
            history,
            game_setup.model_setup.model,
            game_setup.model_setup.temperature,
        )
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
