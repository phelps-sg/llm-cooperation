from __future__ import annotations

import logging
from typing import List, Tuple

from openai_pygenerator import Completion, is_assistant_role, user_message

from llm_cooperation import CT, PT, RT
from llm_cooperation.gametypes.repeated import (
    ChoiceExtractor,
    Choices,
    GameState,
    PayoffFunction,
    ResultForRound,
    RoundsSetup,
    Scores,
    Strategy,
)

logger = logging.getLogger(__name__)


def analyse_round(
    i: int,
    conversation: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
) -> Tuple[Scores, Choices[CT]]:
    assert is_assistant_role(conversation[i * 2])
    ai_choice = extract_choice(conversation[i * 2])
    user_choice = extract_choice(conversation[i * 2 + 1])
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice)
    return Scores(user, ai), Choices(user_choice, ai_choice)


def analyse_rounds(
    history: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
) -> List[ResultForRound]:
    num_messages = len(history)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")
    return [
        analyse_round(i, history, payoffs, extract_choice)
        for i in range(num_messages // 2)
    ]


def next_round(
    partner_strategy: Strategy[CT], state: GameState[CT, PT, RT]
) -> List[Completion]:
    ai_choice = state.game_setup.extract_choice(state.messages[-2])
    user_choice = partner_strategy(state)
    ai_payoff, user_payoff = state.game_setup.payoffs(ai_choice, user_choice)
    return [
        user_message(
            f"Your partner chose {user_choice.description} in that round."
            f"You earned {ai_payoff} and your partner earned {user_payoff}."
            """Now we will move on the next round.
What is your choice for the next round?"""
        )
    ]


rounds_setup = RoundsSetup(analyse_round=analyse_round, analyse_rounds=analyse_rounds)
