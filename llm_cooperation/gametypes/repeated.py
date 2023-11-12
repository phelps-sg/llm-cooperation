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


# @dataclass(frozen=True)
# class RoundsSetup:
#     analyse_round: RoundAnalyser
#     analyse_rounds: RoundsAnalyser


@dataclass(frozen=True)
class GameSetup(Generic[CT, RT]):
    num_rounds: int
    generate_instruction_prompt: PromptGenerator[RT]
    next_round: RoundGenerator[CT, RT]
    analyse_rounds: RoundsAnalyser[CT]
    payoffs: PayoffFunction[CT]
    extract_choice: ChoiceExtractor[CT]
    model_setup: ModelSetup


@dataclass(frozen=True)
class MeasurementSetup(Generic[CT]):
    num_replications: int
    compute_freq: CooperationFrequencyFunction[CT]
    choose_participant_condition: Callable[[], Settings]


@dataclass(frozen=True)
class GameState(Generic[CT, RT]):
    messages: List[Completion]
    round: int
    game_setup: GameSetup[CT, RT]
    participant_condition: Settings


class ChoiceExtractor(Protocol[CT_co]):
    def __call__(
        self, participant_condition: Settings, completion: Completion, **kwargs: bool
    ) -> CT_co:
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
ResultForRound = Tuple[Scores, Choices[CT]]
RoundsAnalyser = Callable[
    [List[Completion], PayoffFunction, ChoiceExtractor[CT], Settings],
    List[ResultForRound[CT]],
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
            partner_strategy,
            GameState(messages, i, game_setup, participant_condition),
            role_prompt,
        )
        messages += partner_response
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: PayoffFunction[CT_contra],
    extract_choice: ChoiceExtractor[CT],
    analyse_rounds: RoundsAnalyser[CT],
    participant_condition: Settings,
) -> Tuple[Scores, List[Choices[CT]]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    results = analyse_rounds(
        conversation, payoffs, extract_choice, participant_condition
    )
    user_score = sum((scores.user for scores, _ in results))
    ai_score = sum((scores.ai for scores, _ in results))
    return Scores(user_score, ai_score), [choices for _, choices in results]


def analyse(
    conversation: List[Completion],
    payoffs: PayoffFunction[CT_contra],
    extract_choice: ChoiceExtractor[CT],
    compute_freq: CooperationFrequencyFunction[CT],
    analyse_rounds: RoundsAnalyser,
    participant_condition: Settings,
) -> Tuple[float, float, Optional[List[Choices[CT]]], List[str], Settings]:
    try:
        history = transcript(conversation)
        result: Tuple[Scores, List[Choices[CT]]] = compute_scores(
            list(conversation),
            payoffs,
            extract_choice,
            analyse_rounds,
            participant_condition,
        )
        scores, choices = result
        freq = compute_freq(choices)
        return scores.ai, freq, choices, history, participant_condition
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)], dict()


def generate_replications(
    participant: RT,
    partner_strategy: Strategy[CT],
    measurement_setup: MeasurementSetup[CT],
    game_setup: GameSetup[CT, RT],
) -> Iterable[Tuple[float, float, Optional[List[Choices[CT]]], List[str], Settings]]:
    # pylint: disable=R0801
    for __i__ in range(measurement_setup.num_replications):
        try:
            participant_condition = measurement_setup.choose_participant_condition()
            conversation = play_game(
                partner_strategy=partner_strategy,
                participant_condition=participant_condition,
                game_setup=game_setup,
                role_prompt=participant,
            )
            yield analyse(
                conversation,
                game_setup.payoffs,
                game_setup.extract_choice,
                measurement_setup.compute_freq,
                game_setup.analyse_rounds,
                participant_condition,
            )
        except ValueError as ex:
            logger.exception(ex)
            yield (np.nan, np.nan, None, [str(ex)], dict())


def run_experiment(
    ai_participants: Dict[Group, List[RT]],
    partner_conditions: Dict[str, Strategy[CT]],
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
        for strategy_name, strategy_fn in partner_conditions.items()
        for score, freq, choices, history, participant_condition in generate_replications(
            participant=participant,
            partner_strategy=strategy_fn,
            measurement_setup=measurement_setup,
            game_setup=game_setup,
        )
    )
