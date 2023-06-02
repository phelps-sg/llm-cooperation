import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openai_pygenerator import (
    Completion,
    Conversation,
    History,
    gpt_completions,
    transcript,
    user_message,
)

Group = Enum(
    "Group",
    ["Cooperative", "Competitive", "Altruistic", "Selfish", "Mixed", "Control"],
)

# pylint: disable=line-too-long
AI_PARTICIPANTS = {
    # Competitive Prompts:
    Group.Cooperative: [
        "You are a helpful and cooperative individual who values teamwork and always seeks the best outcome for all parties involved.",  # noqa: E501
        "You are a diplomatic negotiator, skilled at finding mutually beneficial solutions and fostering collaboration between diverse stakeholders.",  # noqa: E501
        "You are a community organizer who believes in the power of collective action and works tirelessly to bring people together for the greater good.",  # noqa: E501
    ],
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
        "",
    ],
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


class DilemmaChoice(Enum):
    C = auto()
    D = auto()


Choice = DilemmaChoice


@dataclass
class Scores:
    user: int
    ai: int


@dataclass
class Choices:
    user: Choice
    ai: Choice


Strategy = Callable[[List[Completion]], Choice]
ResultRow = Tuple[Group, str, str, int, float, List[Choices], List[str]]


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
    return DilemmaChoice.D


def strategy_cooperate(_history: History) -> DilemmaChoice:
    return DilemmaChoice.C


def strategy_tit_for_tat(
    history: List[Completion], initial_choice: DilemmaChoice = DilemmaChoice.C
) -> DilemmaChoice:
    if len(history) == 2:
        return initial_choice
    ai_choice = extract_choice(history[-2])
    if ai_choice == DilemmaChoice.C:
        return DilemmaChoice.C
    else:
        return DilemmaChoice.D


def strategy_t4t_defect(history: List[Completion]) -> DilemmaChoice:
    return strategy_tit_for_tat(history, initial_choice=DilemmaChoice.D)


def strategy_t4t_cooperate(history: List[Completion]) -> DilemmaChoice:
    return strategy_tit_for_tat(history, initial_choice=DilemmaChoice.C)


def move_as_str(move: DilemmaChoice) -> str:
    if move == DilemmaChoice.D:
        return "Project Blue"
    elif move == DilemmaChoice.C:
        return "Project Green"
    raise ValueError(f"Invalid choice {move}")


def run_single_game(
    num_rounds: int,
    user_strategy: Strategy,
    generate_instruction_prompt: Callable[[int], str],
    role_prompt: str,
) -> Conversation:
    messages = [user_message(role_prompt + generate_instruction_prompt(num_rounds))]
    for _round in range(num_rounds):
        completion = gpt_completions(messages)
        messages += completion
        user_choice = move_as_str(user_strategy(messages))
        messages += [
            user_message(
                f"""Your partner chose {user_choice} in that round.  Now we will move on the next round.
What is your choice for the next round?"""
            )
        ]
    return messages


def extract_choice(
    completion: Completion, regex: str = r"project (blue|green)"
) -> DilemmaChoice:
    logger.debug("completion = %s", completion)
    lower = completion["content"].lower().strip()
    choice_match = re.search(regex, lower)
    if choice_match:
        choice = choice_match.group(1)
        if choice == "green":
            return DilemmaChoice.C
        elif choice == "blue":
            return DilemmaChoice.D
    raise ValueError(f"Could not match choice in {completion}")


def payoffs(
    player1: DilemmaChoice, player2: DilemmaChoice, payoff_matrix: NDArray
) -> Tuple[int, int]:
    def i(m: DilemmaChoice) -> int:
        return m.value - 1

    return (
        payoff_matrix[i(player1), i(player2)],
        payoff_matrix.T[i(player1), i(player2)],
    )


def compute_scores(
    conversation: List[Completion], payoff_matrix: NDArray = PAYOFFS_PD
) -> Tuple[Scores, List[Choices]]:
    conversation = conversation[1:]
    num_messages = len(conversation)
    if num_messages % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")

    def analyse_round(i: int) -> Tuple[Scores, Choices]:
        assert conversation[i * 2]["role"] == "assistant"
        ai_choice = extract_choice(conversation[i * 2])
        user_choice = extract_choice(conversation[i * 2 + 1])
        logger.debug("user_choice = %s", user_choice)
        logger.debug("ai_choice = %s", ai_choice)
        user, ai = payoffs(user_choice, ai_choice, payoff_matrix)
        return Scores(user, ai), Choices(user_choice, ai_choice)

    rounds = [analyse_round(i) for i in range(num_messages // 2)]
    user_score = sum((scores.user for scores, _ in rounds))
    ai_score = sum((scores.ai for scores, _ in rounds))
    return Scores(user_score, ai_score), [choices for _, choices in rounds]


def run_sample(
    prompt: str,
    strategy: Strategy,
    num_samples: int,
    num_rounds: int,
    generate_instruction_prompt: Callable[[int], str],
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
            scores, choices = compute_scores(list(conversation))
            freq = len([c for c in choices if c.ai == DilemmaChoice.C]) / len(choices)
            yield scores.ai, freq, choices, history
        except ValueError:
            yield 0, np.nan, None, history


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    user_conditions: Dict[str, Strategy],
    num_rounds: int,
    generate_instruction_prompt: Callable[[int], str],
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
    )
    df = results_to_df(results)
    filename = "./results/dilemma.pickle"
    logger.info("Experiment complete, saving results to %s", filename)
    df.to_pickle(filename)


if __name__ == "__main__":
    main()
