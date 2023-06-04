import logging
import re
from enum import Enum, auto
from typing import Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from openai_pygenerator import Completion, History

from llm_cooperation import (
    AI_PARTICIPANTS,
    Choice,
    Choices,
    ResultRow,
    Scores,
    run_and_record_experiment,
    run_experiment,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

SAMPLE_SIZE = 30
NUM_ROUNDS = 6

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
    def render(self) -> str:
        return move_as_str(self.value)

    def __init__(self, value: DilemmaEnum):
        self._value = value

    @property
    def value(self) -> DilemmaEnum:
        return self._value


Cooperate = DilemmaChoice(DilemmaEnum.C)
Defect = DilemmaChoice(DilemmaEnum.D)


def get_prompt_pd(n: int) -> str:
    cooperate = """'project green'"""
    defect = """'project blue'"""

    def payout(amount: int) -> str:
        return f"${amount}.00"

    return f"""
This is a study of investment choices in different situations.
You will be making your choices with an investment partner
who is another person. She is in the next room waiting.
In a minute I will take you there yo meet your investment partner.
You and your partner will be choosing between two investment projects and
you will do so for a total of {n} rounds, where in each round both you and
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
but you are not allowed to communicate with your partner apart from informing of them of your choice.
Here is your investment partner.
What is your first choice?
Choice: [{defect} | {cooperate}]
"""


def strategy_defect(_history: History) -> DilemmaChoice:
    return Defect


def strategy_cooperate(_history: History) -> DilemmaChoice:
    return Cooperate


def strategy_tit_for_tat(
    history: List[Completion], initial_choice: DilemmaChoice = Cooperate
) -> DilemmaChoice:
    if len(history) == 2:
        return initial_choice
    ai_choice = extract_choice(history[-2])
    if ai_choice == Cooperate:
        return Cooperate
    else:
        return Defect


def strategy_t4t_defect(history: List[Completion]) -> DilemmaChoice:
    return strategy_tit_for_tat(history, initial_choice=Defect)


def strategy_t4t_cooperate(history: List[Completion]) -> DilemmaChoice:
    return strategy_tit_for_tat(history, initial_choice=Cooperate)


def move_as_str(move: DilemmaEnum) -> str:
    if move == DilemmaEnum.D:
        return "Project Blue"
    elif move == DilemmaEnum.C:
        return "Project Green"
    raise ValueError(f"Invalid choice {move}")


def extract_choice(
    completion: Completion, regex: str = r"project (blue|green)"
) -> DilemmaChoice:
    logger.debug("completion = %s", completion)
    lower = completion["content"].lower().strip()
    choice_match = re.search(regex, lower)
    if choice_match:
        choice = choice_match.group(1)
        if choice == "green":
            return Cooperate
        elif choice == "blue":
            return Defect
    raise ValueError(f"Could not match choice in {completion}")


def payoffs(
    player1: DilemmaChoice, player2: DilemmaChoice, payoff_matrix: NDArray
) -> Tuple[int, int]:
    def i(m: DilemmaChoice) -> int:
        return m.value.value - 1

    return (
        payoff_matrix[i(player1), i(player2)],
        payoff_matrix.T[i(player1), i(player2)],
    )


def analyse_round_pd(
    i: int, conversation: List[Completion], payoff_matrix: NDArray = PAYOFFS_PD
) -> Tuple[Scores, Choices]:
    assert conversation[i * 2]["role"] == "assistant"
    ai_choice = extract_choice(conversation[i * 2])
    user_choice = extract_choice(conversation[i * 2 + 1])
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice, payoff_matrix)
    return Scores(user, ai), Choices(user_choice, ai_choice)


def compute_freq_pd(choices: List[Choices]) -> float:
    return len([c for c in choices if c.ai == Cooperate]) / len(choices)


def run_experiment_pd() -> Iterable[ResultRow]:
    return run_experiment(
        ai_participants=AI_PARTICIPANTS,
        user_conditions={
            "unconditional cooperate": strategy_cooperate,
            "unconditional defect": strategy_defect,
            "tit for tat C": strategy_t4t_cooperate,
            "tit for tat D": strategy_t4t_defect,
        },
        num_rounds=NUM_ROUNDS,
        num_samples=SAMPLE_SIZE,
        generate_instruction_prompt=get_prompt_pd,
        analyse_round=analyse_round_pd,
        compute_freq=compute_freq_pd,
    )


if __name__ == "__main__":
    run_and_record_experiment("dilemma", run_experiment_pd)
