import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import List

import numpy as np
from openai_pygenerator import Completion

from llm_cooperation import ChainOfThoughtCondition, ModelSetup, Payoffs
from llm_cooperation.experiments import AI_PARTICIPANTS, run_and_record_experiment
from llm_cooperation.gametypes import simultaneous
from llm_cooperation.gametypes.repeated import (
    Choices,
    GameSetup,
    GameState,
    MeasurementSetup,
    RepeatedGameResults,
    run_experiment,
)

NUM_ROUNDS: int = 6

T = 7
R = 5
P = 3
S = 0

assert T > R > P > S
assert 2 * R > T + S

PAYOFFS_PD = np.array([[R, S], [T, P]])

COLOR_COOPERATE = "Green"
COLOR_DEFECT = "Blue"

logger = logging.getLogger(__name__)


class DilemmaEnum(Enum):
    C = auto()
    D = auto()


@dataclass
class DilemmaChoice:
    value: DilemmaEnum

    @property
    def description(self) -> str:
        return move_as_str(self.value)

    @property
    def as_int(self) -> int:
        return self.value.value


Cooperate = DilemmaChoice(DilemmaEnum.C)
Defect = DilemmaChoice(DilemmaEnum.D)


def get_prompt_pd(condition: ChainOfThoughtCondition, role_prompt: str) -> str:
    cooperate = f"""'Project {COLOR_COOPERATE}'"""
    defect = f"""'Project {COLOR_DEFECT}'"""
    choice_template = f"""Choice: [{defect} | {cooperate}]"""

    def payout(amount: int) -> str:
        return f"${amount}.00"

    def round_instructions() -> str:
        if condition:
            return f"""
For each round, give your answer in the format below on two separate lines:
Explanation: [why I made my choice]
{choice_template}"""
        else:
            return f"""
    For each round, state your choice without explanation in the format below:
    {choice_template}"""

    return f"""{role_prompt}
This is a study of investment choices in different situations.
You will be making your choices with an investment partner
who is another person. She is in the next room waiting.
In a minute I will take you there to meet your investment partner.
You and your partner will be choosing between two investment projects and
you will do so for a total of {NUM_ROUNDS} rounds, where in each round both you and
your partner will choose a project. The object is to earn as much money as possible.
If you and your partner choose {cooperate} you will both earn {payout(R)}.
If you choose {defect} but your partner chooses {cooperate} then you
will receive {payout(T)} and your partner will receive {payout(S)}.  If you both
choose {defect} you will each receive {payout(P)}. If your partner
chooses {defect} and you choose {cooperate} then you will
receive {payout(S)} and your partner will receive {payout(T)}.
I will tell you what your partner chooses in subsequent prompts,
but you will make your choice ahead of your partner telling me your choice.
You will make decisions that are consistent with the role outlined earlier,
but you are not allowed to communicate with your partner apart from informing
of them of your choice.
{round_instructions()}
Here is your investment partner. What is your choice in the first round?
"""


# pylint: disable=unused-argument
def strategy_defect(
    state: GameState[DilemmaChoice, ChainOfThoughtCondition, str], **__kwargs__: bool
) -> DilemmaChoice:
    return Defect


# pylint: disable=unused-argument
def strategy_cooperate(
    state: GameState[DilemmaChoice, ChainOfThoughtCondition, str], **__kwargs__: bool
) -> DilemmaChoice:
    return Cooperate


def strategy_t4t(
    initial_choice: DilemmaChoice,
    state: GameState[DilemmaChoice, ChainOfThoughtCondition, str],
    **__kwargs__: bool,
) -> DilemmaChoice:
    if len(state.messages) == 2:
        return initial_choice
    previous_message = state.messages[-1]
    logger.debug("previous_message = %s", previous_message)
    ai_choice = extract_choice_pd(previous_message)
    logger.debug("ai_choice = %s", ai_choice)
    if ai_choice == Cooperate:
        return Cooperate
    else:
        return Defect


strategy_t4t_defect = partial(strategy_t4t, Defect)
strategy_t4t_cooperate = partial(strategy_t4t, Cooperate)


def move_as_str(move: DilemmaEnum) -> str:
    if move == DilemmaEnum.D:
        return f"Project {COLOR_DEFECT}"
    elif move == DilemmaEnum.C:
        return f"Project {COLOR_COOPERATE}"
    raise ValueError(f"Invalid choice {move}")


def choice_from_str(choice: str) -> DilemmaChoice:
    if choice == COLOR_COOPERATE.lower():
        return Cooperate
    elif choice == COLOR_DEFECT.lower():
        return Defect
    else:
        raise ValueError(f"Cannot determine choice from {choice}")


def extract_choice_pd(completion: Completion, **__kwargs__: bool) -> DilemmaChoice:
    regex: str = rf".*project ({COLOR_COOPERATE}|{COLOR_DEFECT})".lower()
    choice_regex: str = f"choice:{regex}"
    logger.debug("completion = %s", completion)
    lower = completion["content"].lower().strip()

    def matched_choice(m: re.Match) -> DilemmaChoice:
        return choice_from_str(m.group(1))

    match = re.search(choice_regex, lower)
    if match is not None:
        return matched_choice(match)
    else:
        match = re.search(regex, lower)
        if match is not None:
            return matched_choice(match)
    raise ValueError(f"Cannot determine choice from {completion}")


def payoffs_pd(player1: DilemmaChoice, player2: DilemmaChoice) -> Payoffs:
    def i(m: DilemmaChoice) -> int:
        return m.as_int - 1

    return (
        PAYOFFS_PD[i(player1), i(player2)],
        PAYOFFS_PD.T[i(player1), i(player2)],
    )


def compute_freq_pd(choices: List[Choices[DilemmaChoice]]) -> float:
    return len([c for c in choices if c.ai == Cooperate]) / len(choices)


def run(model_setup: ModelSetup, sample_size: int) -> RepeatedGameResults:
    game_setup: GameSetup[DilemmaChoice, ChainOfThoughtCondition, str] = GameSetup(
        num_rounds=NUM_ROUNDS,
        generate_instruction_prompt=get_prompt_pd,
        payoffs=payoffs_pd,
        extract_choice=extract_choice_pd,
        next_round=simultaneous.next_round,
        analyse_rounds=simultaneous.analyse_rounds,
        model_setup=model_setup,
    )
    measurement_setup: MeasurementSetup[DilemmaChoice] = MeasurementSetup(
        num_samples=sample_size,
        compute_freq=compute_freq_pd,
    )
    return run_experiment(
        ai_participants=AI_PARTICIPANTS,
        partner_conditions={
            "unconditional cooperate": strategy_cooperate,
            "unconditional defect": strategy_defect,
            "tit for tat C": strategy_t4t_cooperate,
            "tit for tat D": strategy_t4t_defect,
        },
        participant_conditions={"chain-of-thought": True, "no-chain-of-thought": False},
        measurement_setup=measurement_setup,
        game_setup=game_setup,
    )


if __name__ == "__main__":
    run_and_record_experiment("dilemma", run)
