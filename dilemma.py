import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List, Tuple, Callable, Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import gpt

Group = Enum(
    "Group",
    ["Competitive", "Altruistic", "Selfish", "Mixed", "Control"],
)

# pylint: disable=line-too-long
AI_PARTICIPANTS = {

    # Competitive Prompts:
    Group.Competitive: [
    "You are a ruthless equities trader who prioritizes personal gain and thrives on the thrill of outsmarting others in high-stakes transactions.",  # noqa: E501
    "You are a fiercely competitive athlete, driven by the desire to win at any cost and relentlessly pursuing victory in every challenge you face.",  # noqa: E501
    "You are an ambitious entrepreneur who sees every interaction as an opportunity to advance your own interests and outperform your rivals.",  # noqa: E501
        ],

    # Altruistic Prompts:
    Group.Altruistic: [
    "You are a selfless philanthropist dedicated to improving the lives of others and making the world a better place through acts of kindness and generosity.",  # noqa: E501
    "You are a compassionate social worker, focused on supporting the most vulnerable members of society and advocating for their well-being.",  # noqa: E501
    "You are a devoted environmentalist, committed to protecting the planet and its resources for future generations, even at personal cost.",  # noqa: E501
        ],

    # Self-interested Prompts:
    Group.Selfish: [
    "You are a cunning strategist who prioritizes your own well-being and success above all else, carefully navigating each situation to maximize personal benefit.",  # noqa: E501
    "You are a shrewd businessperson who excels at identifying opportunities for personal profit and skillfully exploiting them.",  # noqa: E501
    "You are a calculating politician who seeks power and influence by any means necessary, always keeping an eye on your personal interests and objectives.",  # noqa: E501
        ],

    # Mixed-motivation Prompts:
    Group.Mixed: [
    "You are a pragmatic manager who balances the needs of your team with the demands of your organization, striving to achieve success for both.",  # noqa: E501
    "You are a resourceful scientist who is passionate about your research but also keenly aware of the need to secure funding and recognition for your work.",  # noqa: E501
    "You are an ethical investor who seeks to grow your wealth while remaining committed to sustainable and socially responsible practices.",  # noqa: E501
        ],

    # Controls
    Group.Control: [
    "You are a participant in a psychology experiment.",
    "You are a helpful assistant.",
    ""
    ]
}

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


class Choice(Enum):
    C = auto()
    D = auto()


@dataclass
class Scores:
    user: int
    ai: int


@dataclass
class Choices:
    user: Choice
    ai: Choice


Strategy = Callable[[List[gpt.Completion]], Choice]
Results = Dict[Tuple[Group, str, str], Tuple[float, float, float, float, int]]


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


def strategy_tit_for_tat(
    history: List[gpt.Completion], initial_choice: Choice = Choice.C
) -> Choice:
    if len(history) == 3:
        return initial_choice
    ai_choice = extract_choice(history[-2])
    if ai_choice == Choice.C:
        return Choice.C
    else:
        return Choice.D


def strategy_t4t_defect(history: List[gpt.Completion]):
    return strategy_tit_for_tat(history, initial_choice=Choice.D)


def strategy_t4t_cooperate(history: List[gpt.Completion]):
    return strategy_tit_for_tat(history, initial_choice=Choice.C)


def move_as_str(move: Choice) -> str:
    if move == Choice.D:
        return "Project Blue"
    elif move == Choice.C:
        return "Project Green"
    raise ValueError(f"Invalid choice {move}")


def run_prisoners_dilemma(
    num_rounds: int = NUM_ROUNDS,
    user_strategy: Strategy = strategy_defect,
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


def extract_choice(
    completion: gpt.Completion, regex: str = r"project (blue|green)"
) -> Choice:
    logger.debug("completion = %s", completion)
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


def run_sample(prompt: str, strategy: Strategy, n: int) -> Iterable[Tuple[int, float]]:
    for _i in range(n):
        try:
            conversation = run_prisoners_dilemma(
                role_prompt=prompt, user_strategy=strategy
            )
            scores, choices = compute_scores(list(conversation))
            freq = len([c for c in choices if c.ai == Choice.C]) / len(choices)
            yield scores.ai, freq
        except ValueError:
            yield 0, np.nan


def results_as_df(results_by_condition: Results) -> pd.DataFrame:
    df = pd.DataFrame(results_by_condition).transpose()
    df.columns = pd.Index(
        [
            "score (mean)",
            "score (std)",
            "cooperation frequency (mean)",
            "cooperation frequency (std)",
            "N",
        ]
    )
    return df


def mean(values: NDArray) -> float:
    return float(np.nanmean(values, dtype=float))


def std(values: NDArray) -> float:
    return float(np.nanstd(values, dtype=float))


def print_report(results_by_condition: Results) -> None:
    print()
    for (_group, prompt, strategy_name), (
        mean_score,
        _std_score,
        mean_freq,
        _std_freq,
        n,
    ) in results_by_condition.items():
        print(f"{prompt} playing {strategy_name}")
        print(f"Sample size = {n}")
        print(f"Mean score = {mean_score}")
        print(f"Mean cooperation frequency = {round(mean_freq, 2)}")
        print()


def run_experiment(
    ai_participants: Dict[Group, List[str]], user_conditions: Dict[str, Strategy]
) -> None:
    results_by_condition: Results = {}
    for group, prompts in ai_participants.items():
        for prompt in prompts:
            for strategy_name, strategy_fn in user_conditions.items():
                results = list(run_sample(prompt, strategy_fn, SAMPLE_SIZE))
                frequencies = np.array([freq for _score, freq in results])
                scores = np.array([score for score, _freq in results])
                n = len([x for _, x in results if not np.isnan(x)])
                results_by_condition[(group, prompt, strategy_name)] = (
                    mean(scores),
                    std(scores),
                    mean(frequencies),
                    std(frequencies),
                    n,
                )
    print_report(results_by_condition)
    df = results_as_df(results_by_condition)
    df.to_pickle("results.pickle")


def main() -> None:
    run_experiment(
        ai_participants=AI_PARTICIPANTS,
        user_conditions={
            "unconditional cooperate": strategy_cooperate,
            "unconditional defect": strategy_defect,
            "tit for tat C": strategy_t4t_cooperate,
            "tit for tat D": strategy_t4t_defect,
        },
    )


if __name__ == "__main__":
    main()
