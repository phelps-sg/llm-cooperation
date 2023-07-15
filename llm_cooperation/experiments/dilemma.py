import logging
import re
from enum import Enum, auto
from typing import List

import numpy as np
from openai_pygenerator import Completion

from llm_cooperation import ChainOfThoughtCondition, Choice, ModelSetup, Payoffs
from llm_cooperation.experiments import AI_PARTICIPANTS, run_and_record_experiment
from llm_cooperation.gametypes import simultaneous
from llm_cooperation.gametypes.repeated import (
    Choices,
    GameSetup,
    GameState,
    MeasurementSetup,
    RepeatedGameResults,
    Strategy,
    run_experiment,
)

SAMPLE_SIZE: int = 30
NUM_ROUNDS: int = 6

T = 7
R = 5
P = 3
S = 0

assert T > R > P > S
assert 2 * R > T + S

PAYOFFS_PD = np.array([[R, S], [T, P]])

logger = logging.getLogger(__name__)


class DilemmaEnum(Enum):
    C = auto()
    D = auto()


class DilemmaChoice(Choice):
    def __init__(self, value: DilemmaEnum):
        self._value = value

    @property
    def description(self) -> str:
        return move_as_str(self.value)

    @property
    def value(self) -> DilemmaEnum:
        return self._value

    @property
    def as_int(self) -> int:
        return self.value.value


Cooperate = DilemmaChoice(DilemmaEnum.C)
Defect = DilemmaChoice(DilemmaEnum.D)


def get_prompt_pd(condition: ChainOfThoughtCondition, role_prompt: str) -> str:
    cooperate = """'project green'"""
    defect = """'project blue'"""
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
In a minute I will take you there yo meet your investment partner.
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
    state: GameState[DilemmaChoice, ChainOfThoughtCondition], **_kwargs: bool
) -> DilemmaChoice:
    return Defect


# pylint: disable=unused-argument
def strategy_cooperate(
    state: GameState[DilemmaChoice, ChainOfThoughtCondition], **_kwargs: bool
) -> DilemmaChoice:
    return Cooperate


def make_tit_for_tat(
    initial_choice: DilemmaChoice = Cooperate,
) -> Strategy[DilemmaChoice]:
    def tit_for_tat(
        state: GameState[DilemmaChoice, ChainOfThoughtCondition], **_kwargs: bool
    ) -> DilemmaChoice:
        if len(state.messages) == 2:
            return initial_choice
        ai_choice = extract_choice_pd(state.messages[-2])
        if ai_choice == Cooperate:
            return Cooperate
        else:
            return Defect

    return tit_for_tat


strategy_t4t_defect = make_tit_for_tat(initial_choice=Defect)
strategy_t4t_cooperate = make_tit_for_tat(initial_choice=Cooperate)


def move_as_str(move: DilemmaEnum) -> str:
    if move == DilemmaEnum.D:
        return "Project Blue"
    elif move == DilemmaEnum.C:
        return "Project Green"
    raise ValueError(f"Invalid choice {move}")


def choice_from_str(choice: str) -> DilemmaChoice:
    if choice == "green":
        return Cooperate
    elif choice == "blue":
        return Defect
    else:
        raise ValueError(f"Cannot determine choice from {choice}")


def extract_choice_pd(completion: Completion, **_kwargs: bool) -> DilemmaChoice:
    regex: str = r".*project (blue|green)"
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


def run(model_setup: ModelSetup, sample_size: int = SAMPLE_SIZE) -> RepeatedGameResults:
    game_setup: GameSetup[DilemmaChoice, ChainOfThoughtCondition] = GameSetup(
        num_rounds=NUM_ROUNDS,
        generate_instruction_prompt=get_prompt_pd,
        payoffs=payoffs_pd,
        extract_choice=extract_choice_pd,
        next_round=simultaneous.next_round,
        rounds=simultaneous.rounds_setup,
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
