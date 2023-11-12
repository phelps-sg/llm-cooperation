import logging
from typing import List

from openai_pygenerator import Completion

from llm_cooperation import CT, Settings
from llm_cooperation.gametypes.repeated import (
    ChoiceExtractor,
    Choices,
    PayoffFunction,
    ResultForRound,
    Scores,
)

logger = logging.getLogger(__name__)


def analyse_rounds(
    history: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    participant_condition: Settings,
) -> List[ResultForRound[CT]]:
    num_messages = len(history)
    return [
        analyse_round(i, history, payoffs, extract_choice, participant_condition)
        for i in range(num_messages - 1)
    ]


def analyse_round(
    i: int,
    conversation: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    participant_condition: Settings,
) -> ResultForRound[CT]:
    """
        Analyse round of this form:


         : AI: 		Propose: $10               True
    0 -> : USER: 	Accept / Propose: $10      False
         : AI: 	    Accept / Propose: $10      True
    1 -> : USER: 	Accept / Propose: $10      False
    """

    users_turn = (i % 2) > 0
    ai_index = i + 1 if users_turn else i
    user_index = i if users_turn else i + 1
    ai_completion = conversation[ai_index]
    user_completion = conversation[user_index]
    ai_choice = extract_choice(
        participant_condition, ai_completion, proposer=not users_turn
    )
    user_choice = extract_choice(
        participant_condition, user_completion, proposer=users_turn
    )
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice)
    return Scores(user, ai), Choices(user_choice, ai_choice)
