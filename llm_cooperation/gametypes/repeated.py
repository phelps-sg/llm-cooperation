from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, transcript

from llm_cooperation import (
    CT,
    RT,
    CT_co,
    CT_contra,
    Grid,
    Group,
    ModelSetup,
    Payoffs,
    Results,
    Settings,
)
from llm_cooperation.gametypes import PromptGenerator, start_game

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Choices(Generic[CT]):
    user: CT
    ai: CT


@dataclass(frozen=True)
class Scores:
    user: float
    ai: float


@dataclass(frozen=True)
class RoundsSetup:
    analyse_round: RoundAnalyser
    analyse_rounds: RoundsAnalyser


@dataclass(frozen=True)
class GameSetup(Generic[CT, RT]):
    num_rounds: int
    generate_instruction_prompt: PromptGenerator[RT]
    next_round: RoundGenerator
    analyse_rounds: RoundsAnalyser
    payoffs: PayoffFunction[CT]
    extract_choice: ChoiceExtractor[CT]
    model_setup: ModelSetup
    participant_condition_sampling: Callable[[Grid], Iterable[Settings]]


@dataclass(frozen=True)
class MeasurementSetup(Generic[CT]):
    num_samples: int
    compute_freq: CooperationFrequencyFunction[CT]


@dataclass(frozen=True)
class GameState(Generic[CT, RT]):
    messages: List[Completion]
    round: int
    game_setup: GameSetup[CT, RT]


class ChoiceExtractor(Protocol[CT_co]):
    def __call__(self, completion: Completion, **kwargs: bool) -> CT_co:
        ...


class Strategy(Protocol[CT_co]):
    def __call__(self, state: GameState, **kwargs: bool) -> CT_co:
        ...


RoundGenerator = Callable[[Strategy[CT], GameState[CT, RT], RT], List[Completion]]


class CooperationFrequencyFunction(Protocol[CT]):
    def __call__(self, choices: List[Choices[CT]]) -> float:
        ...


class PayoffFunction(Protocol[CT_contra]):
    def __call__(self, player1: CT_contra, player2: CT_contra) -> Payoffs:
        ...


# PromptGenerator = Callable[[ParticipantCondition, str], str]
ResultForRound = Tuple[Scores, Choices]
RoundAnalyser = Callable[
    [int, List[Completion], PayoffFunction, ChoiceExtractor], ResultForRound
]
RoundsAnalyser = Callable[
    [List[Completion], PayoffFunction, ChoiceExtractor], List[ResultForRound]
]

ResultRepeatedGame = Tuple[
    Group,
    RT,
    Settings,
    str,
    float,
    float,
    Optional[List[Choices]],
    List[str],
    str,
    float,
]


class RepeatedGameResults(Results):
    def __init__(self, rows: Iterable[ResultRepeatedGame]):
        self._rows: Iterable[ResultRepeatedGame] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    str(group),
                    participant,
                    condition,
                    strategy,
                    score,
                    freq,
                    choices,
                    history,
                    model,
                    temp,
                )
                # pylint: disable=line-too-long
                for group, participant, condition, strategy, score, freq, choices, history, model, temp in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Participant Condition",
                "Partner Condition",
                "Score",
                "Cooperation frequency",
                "Choices",
                "Transcript",
                "Model",
                "Temperature",
            ],
        )


def play_game(
    role_prompt: RT,
    participant_condition: Settings,
    partner_strategy: Strategy[CT],
    game_setup: GameSetup[CT, RT],
) -> List[Completion]:
    gpt_completions, messages = start_game(
        game_setup.generate_instruction_prompt,
        game_setup.model_setup,
        participant_condition,
        role_prompt,
    )
    for i in range(game_setup.num_rounds):
        completion = gpt_completions(messages, 1)
        messages += completion
        partner_response = game_setup.next_round(
            partner_strategy, GameState(messages, i, game_setup), role_prompt
        )
        messages += partner_response
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: PayoffFunction[CT_contra],
    extract_choice: ChoiceExtractor[CT],
    analyse_rounds: RoundsAnalyser,
) -> Tuple[Scores, List[Choices[CT]]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    results = analyse_rounds(conversation, payoffs, extract_choice)
    user_score = sum((scores.user for scores, _ in results))
    ai_score = sum((scores.ai for scores, _ in results))
    return Scores(user_score, ai_score), [choices for _, choices in results]


def analyse(
    conversation: List[Completion],
    payoffs: PayoffFunction[CT_contra],
    extract_choice: ChoiceExtractor[CT],
    compute_freq: CooperationFrequencyFunction[CT],
    analyse_rounds: RoundsAnalyser,
) -> Tuple[float, float, Optional[List[Choices[CT]]], List[str]]:
    try:
        history = transcript(conversation)
        result: Tuple[Scores, List[Choices[CT]]] = compute_scores(
            list(conversation), payoffs, extract_choice, analyse_rounds
        )
        scores, choices = result
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_samples(
    participant: RT,
    condition: Settings,
    partner_strategy: Strategy[CT],
    measurement_setup: MeasurementSetup[CT],
    game_setup: GameSetup[CT, RT],
) -> Iterable[Tuple[float, float, Optional[List[Choices[CT]]], List[str]]]:
    # pylint: disable=R0801
    for __i__ in range(measurement_setup.num_samples):
        try:
            conversation = play_game(
                partner_strategy=partner_strategy,
                participant_condition=condition,
                game_setup=game_setup,
                role_prompt=participant,
            )
            yield analyse(
                conversation,
                game_setup.payoffs,
                game_setup.extract_choice,
                measurement_setup.compute_freq,
                game_setup.analyse_rounds,
            )
        except ValueError as ex:
            logger.exception(ex)
            yield (np.nan, np.nan, None, [str(ex)])


def run_experiment(
    ai_participants: Dict[Group, List[RT]],
    partner_conditions: Dict[str, Strategy[CT]],
    participant_conditions: Grid,
    measurement_setup: MeasurementSetup[CT],
    game_setup: GameSetup[CT, RT],
) -> RepeatedGameResults:
    return RepeatedGameResults(
        (
            group,
            participant,
            participant_condition,
            strategy_name,
            score,
            freq,
            choices,
            history,
            game_setup.model_setup.model,
            game_setup.model_setup.temperature,
        )
        for group, participants in ai_participants.items()
        for participant in participants
        for participant_condition in game_setup.participant_condition_sampling(
            participant_conditions
        )
        for strategy_name, strategy_fn in partner_conditions.items()
        for score, freq, choices, history in generate_samples(
            participant=participant,
            partner_strategy=strategy_fn,
            condition=participant_condition,
            measurement_setup=measurement_setup,
            game_setup=game_setup,
        )
    )
