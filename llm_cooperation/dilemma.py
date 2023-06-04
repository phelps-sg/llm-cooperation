from __future__ import annotations

import logging
import re
from enum import Enum, auto
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openai_pygenerator import Completion, History, transcript

from llm_cooperation import (
    AI_PARTICIPANTS,
    Choice,
    Choices,
    Group,
    ResultRow,
    Scores,
    Strategy,
    compute_scores,
    run_single_game,
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
        return move_as_str(self.value())

    def __init__(self, value: DilemmaEnum):
        self._value = value

    def value(self) -> DilemmaEnum:
        return self._value


Cooperate = DilemmaChoice(DilemmaEnum.C)
Defect = DilemmaChoice(DilemmaEnum.D)


def prisoners_dilemma_instructions(n: int) -> str:
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


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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
        return m.value().value - 1

    return (
        payoff_matrix[i(player1), i(player2)],
        payoff_matrix.T[i(player1), i(player2)],
    )


def analyse_round_prisoners_dilemma(
    i: int, conversation: List[Completion], payoff_matrix: NDArray = PAYOFFS_PD
) -> Tuple[Scores, Choices]:
    assert conversation[i * 2]["role"] == "assistant"
    ai_choice = extract_choice(conversation[i * 2])
    user_choice = extract_choice(conversation[i * 2 + 1])
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice, payoff_matrix)
    return Scores(user, ai), Choices(user_choice, ai_choice)


def run_sample(
    prompt: str,
    strategy: Strategy,
    num_samples: int,
    num_rounds: int,
    generate_instruction_prompt: Callable[[int], str],
    analyse_round: Callable[[int, List[Completion]], Tuple[Scores, Choices]],
) -> Iterable[Tuple[int, float, Optional[List[Choices]], List[str]]]:
    for _i in range(num_samples):
        conversation = run_single_game(
            num_rounds=num_rounds,
            role_prompt=prompt,
            user_strategy=strategy,
            generate_instruction_prompt=generate_instruction_prompt,
        )
        history = transcript(conversation)
        try:
            scores, choices = compute_scores(list(conversation), analyse_round)
            freq = len([c for c in choices if c.ai == Cooperate]) / len(choices)
            yield scores.ai, freq, choices, history
        except ValueError:
            yield 0, np.nan, None, history


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    user_conditions: Dict[str, Strategy],
    num_rounds: int,
    generate_instruction_prompt: Callable[[int], str],
    analyse_round: Callable[[int, List[Completion]], Tuple[Scores, Choices]],
) -> Iterable[ResultRow]:
    return (
        (group, prompt, strategy_name, score, freq, choices, history)
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for strategy_name, strategy_fn in user_conditions.items()
        for score, freq, choices, history in run_sample(
            prompt=prompt,
            strategy=strategy_fn,
            num_samples=SAMPLE_SIZE,
            num_rounds=num_rounds,
            generate_instruction_prompt=generate_instruction_prompt,
            analyse_round=analyse_round,
        )
    )


def results_to_df(results: Iterable[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (str(group), prompt, strategy_name, score, freq, choices, history)
            for group, prompt, strategy_name, score, freq, choices, history in results
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


def main() -> None:
    results = run_experiment(
        ai_participants=AI_PARTICIPANTS,
        user_conditions={
            "unconditional cooperate": strategy_cooperate,
            "unconditional defect": strategy_defect,
            "tit for tat C": strategy_t4t_cooperate,
            "tit for tat D": strategy_t4t_defect,
        },
        num_rounds=NUM_ROUNDS,
        generate_instruction_prompt=prisoners_dilemma_instructions,
        analyse_round=analyse_round_prisoners_dilemma,
    )
    df = results_to_df(results)
    filename = "./results/dilemma.pickle"
    logger.info("Experiment complete, saving results to %s", filename)
    df.to_pickle(filename)


if __name__ == "__main__":
    main()
