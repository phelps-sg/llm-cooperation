import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List, Tuple, Callable, Dict

import numpy as np
from numpy.typing import NDArray

import gpt

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


@dataclass
class Scores:
    user: int
    ai: int


class Choice(Enum):
    C = auto()
    D = auto()


@dataclass
class Choices:
    user: Choice
    ai: Choice


def get_prompt(n: int) -> str:
    return f"""
 This is a study of investment choices in different situations.  
You will be making your choices with an investment partner who is another person.  She is in the next room waiting.  
In a minute I will take you there yo meet your investment partner.  
You and your partner will be choosing between two investment projects and you will do so {n} times. 
The object is to earn as much money as possible.  
If you and your partner choose 'project green' you will both earn ${R}.00.  
If you choose "project blue" but your partner chooses "project green" then you will receive ${T}.00 
and your partner will receive ${S}.00.  If you both choose project blue you will each receive ${S}.00
If your partner chooses "project blue" and you choose "project green" then you will receive ${S}.00 and your 
partner will receive ${T}.00.  I will tell you what your partner chooses in subsequent prompts, 
but you will make your choice ahead of your partner telling me your choice. 
Here is your investment partner.
What is your first choice?
Choice: [project blue | project green]
"""


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def strategy_defect(_history: gpt.Conversation) -> Choice:
    return Choice.D


def strategy_cooperate(_history: gpt.Conversation) -> Choice:
    return Choice.C


def strategy_tit_for_tat(history: List[gpt.Completion]) -> Choice:
    ai_choice = extract_choice(history[-2])
    if ai_choice == Choice.C:
        return Choice.C
    else:
        return Choice.D


def move_as_str(move: Choice) -> str:
    if move == Choice.D:
        return "Project Blue"
    elif move == Choice.C:
        return "Project Green"
    raise ValueError(f"Invalid choice {move}")


def run_prisoners_dilemma(
    num_rounds: int = NUM_ROUNDS,
    user_strategy: Callable[[gpt.Conversation], Choice] = strategy_defect,
    role_prompt: str = "You are a participant in a psychology experiment.",
) -> gpt.Conversation:
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": get_prompt(num_rounds)},
    ]
    for _round in range(num_rounds):
        completion = gpt.generate_completions(messages)
        messages += completion
        user_choice = move_as_str(user_strategy(messages))
        messages += [
            {
                "role": "user",
                "content": f"Your partner chose {user_choice}.  What was your choice?",
            }
        ]
    return messages


def transcript(messages: gpt.Conversation) -> Iterable[str]:
    return [r["content"] for r in messages]


def extract_choice(completion: gpt.Completion, regex: str = r"project (blue|green)") -> Choice:
    logger.debug(f"completion = {completion}")
    lower = completion["content"].lower().strip()
    choice_match = re.search(regex, lower)
    if choice_match:
        choice = choice_match.group(1)
        if choice == "green":
            return Choice.C
        elif choice == "blue":
            return Choice.D
    raise ValueError(f"Could not match choice in {completion}")


def payoffs(
    player1: Choice, player2: Choice, payoff_matrix: NDArray
) -> Tuple[int, int]:
    def i(m: Choice) -> int:
        return m.value - 1

    return (
        payoff_matrix[i(player1), i(player2)],
        payoff_matrix.T[i(player1), i(player2)],
    )


def compute_scores(
    conversation: List[gpt.Completion], payoff_matrix=PAYOFFS_PD
) -> Tuple[Scores, List[Choices]]:
    user_score = 0
    ai_score = 0

    conversation = conversation[2:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")

    moves = []

    for i in range(num_messages // 2):
        assert conversation[i * 2]["role"] == "assistant"
        ai_choice = extract_choice(conversation[i * 2])
        user_choice = extract_choice(conversation[i * 2 + 1])
        logger.debug("user_choice = %s", user_choice)
        logger.debug("ai_choice = %s", ai_choice)
        user_payoff, ai_payoff = payoffs(user_choice, ai_choice, payoff_matrix)
        user_score += user_payoff
        ai_score += ai_payoff
        moves.append(Choices(user_choice, ai_choice))

    return Scores(user=user_score, ai=ai_score), moves


def run_sample(prompt: str, n: int) -> Iterable[Tuple[int, float]]:
    for _i in range(n):
        conversation = run_prisoners_dilemma(role_prompt=prompt)
        scores, choices = compute_scores(list(conversation))
        freq = len([c for c in choices if c.ai == Choice.C]) / len(choices)
        yield scores.ai, freq


def main() -> None:
    prompts = [
        "You are a participant in a psychology experiment.",
        "You are a helpful and cooperative individual who values teamwork and always seeks the best outcome for all parties involved.",
        "You are a ruthless equities trader who prioritizes personal gain and thrives on the thrill of outsmarting others in high-stakes transactions.",
        "You are a shrewd businessperson who excels at identifying opportunities for personal profit and skillfully exploiting them.",
    ]
    results_by_prompt = dict()
    for prompt in prompts:
        results = list(run_sample(prompt, SAMPLE_SIZE))
        mean_freq = np.mean([freq for _score, freq in results])
        mean_score = np.mean([score for score, _freq in results])
        results_by_prompt[prompt] = mean_score, mean_freq
    print()
    for prompt, (mean_score, mean_freq) in results_by_prompt.items():
        print(prompt)
        print(f"Mean score = {mean_score}")
        print(f"Mean cooperation frequency = {mean_freq}")
        print()


if __name__ == "__main__":
    main()
