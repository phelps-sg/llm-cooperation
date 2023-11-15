#  MIT License
#
#  Copyright (c) 2023 Steve Phelps
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
from openai_pygenerator import Completion

from llm_cooperation import ModelSetup, Payoffs, Settings, randomized
from llm_cooperation.experiments import (
    AI_PARTICIPANTS,
    CONDITION_CASE,
    Case,
    apply_case_condition,
    run_and_record_experiment,
)
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

CONDITION_LABEL = "label"
CONDITION_LABELS_REVERSED = "labels_reversed"
CONDITION_CHAIN_OF_THOUGHT = "chain_of_thought"
CONDITION_DEFECT_FIRST = "defect_first"
CONDITION_PRONOUN = "pronoun"

logger = logging.getLogger(__name__)


class DilemmaEnum(Enum):
    C = auto()
    D = auto()


@dataclass(frozen=True)
class DilemmaChoice:
    value: DilemmaEnum

    def description(self, participant_condition: Settings) -> str:
        return move_as_str(self.value, participant_condition)

    @property
    def as_int(self) -> int:
        return self.value.value


Cooperate = DilemmaChoice(DilemmaEnum.C)
Defect = DilemmaChoice(DilemmaEnum.D)


class Label(Enum):
    COLORS = "colors"
    NUMBERS = "numbers"
    NUMERALS = "numerals"


class Pronoun(Enum):
    HE = "he"
    SHE = "she"
    THEY = "they"


def labels(condition: Settings) -> Tuple[str, str]:
    value = condition[CONDITION_LABEL]
    result: Optional[List[str]] = None
    if value == Label.COLORS.value:
        result = ["Green", "Blue"]
    elif value == Label.NUMBERS.value:
        result = ["One", "Two"]
    elif value == Label.NUMERALS.value:
        result = ["1", "2"]
    if result is not None:
        if condition[CONDITION_LABELS_REVERSED]:
            return result[1], result[0]
        return result[0], result[1]
    raise ValueError(
        f"Unknown value {condition[CONDITION_LABEL]} for {CONDITION_LABEL}"
    )


def cooperate_label(condition: Settings) -> str:
    return labels(condition)[0]


def defect_label(condition: Settings) -> str:
    return labels(condition)[1]


def get_prompt_pd(condition: Settings, role_prompt: str) -> str:
    return apply_case_condition(condition, get_prompt_base(condition, role_prompt))


def get_choice_template(condition: Settings, cooperate: str, defect: str) -> str:
    choices = (
        f"""{defect} | {cooperate}]"""
        if condition[CONDITION_DEFECT_FIRST]
        else f"""{cooperate} | {defect}"""
    )
    return f"""Choice: [{choices}]"""


def get_pronoun_phrasing(condition: Settings) -> str:
    if condition[CONDITION_PRONOUN] == Pronoun.HE.value:
        return "He is"
    elif condition[CONDITION_PRONOUN] == Pronoun.SHE.value:
        return "She is"
    elif condition[CONDITION_PRONOUN] == Pronoun.THEY.value:
        return "They are"
    raise ValueError(
        f"Invalid value {condition[CONDITION_PRONOUN]} for {CONDITION_PRONOUN}"
    )


def get_prompt_base(condition: Settings, role_prompt: str) -> str:
    cooperate = f"""'Project {cooperate_label(condition)}'"""
    defect = f"""'Project {defect_label(condition)}'"""
    choice_template = get_choice_template(condition, cooperate, defect)
    pronoun_phrase = get_pronoun_phrasing(condition)

    def payout(amount: int) -> str:
        return f"${amount}.00"

    def round_instructions() -> str:
        if condition["chain_of_thought"]:
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
who is another person. {pronoun_phrase} in the next room waiting.
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
    state: GameState[DilemmaChoice, str], **__kwargs__: bool
) -> DilemmaChoice:
    return Defect


# pylint: disable=unused-argument
def strategy_cooperate(
    state: GameState[DilemmaChoice, str], **__kwargs__: bool
) -> DilemmaChoice:
    return Cooperate


def strategy_t4t(
    initial_choice: DilemmaChoice,
    state: GameState[DilemmaChoice, str],
    **__kwargs__: bool,
) -> DilemmaChoice:
    if len(state.messages) == 2:
        return initial_choice
    previous_message = state.messages[-1]
    logger.debug("previous_message = %s", previous_message)
    ai_choice = extract_choice_pd(state.participant_condition, previous_message)
    logger.debug("ai_choice = %s", ai_choice)
    if ai_choice == Cooperate:
        return Cooperate
    else:
        return Defect


strategy_t4t_defect = partial(strategy_t4t, Defect)
strategy_t4t_cooperate = partial(strategy_t4t, Cooperate)


def move_as_str(move: DilemmaEnum, participant_condition: Settings) -> str:
    if move == DilemmaEnum.D:
        return f"Project {defect_label(participant_condition)}"
    elif move == DilemmaEnum.C:
        return f"Project {cooperate_label(participant_condition)}"
    raise ValueError(f"Invalid choice {move}")


def choice_from_str(choice: str, participant_condition: Settings) -> DilemmaChoice:
    logger.debug("participant_condition = %s", participant_condition)
    if choice == cooperate_label(participant_condition).lower():
        return Cooperate
    elif choice == defect_label(participant_condition).lower():
        return Defect
    else:
        raise ValueError(f"Cannot determine choice from {choice}")


def extract_choice_pd(
    participant_condition: Settings, completion: Completion, **__kwargs__: bool
) -> DilemmaChoice:
    logger.debug("participant_condition = %s", participant_condition)
    cooperate = cooperate_label(participant_condition)
    defect = defect_label(participant_condition)
    regex: str = rf".*project\s+({cooperate}|{defect})".lower()
    choice_regex: str = f"choice:{regex}"
    logger.debug("completion = %s", completion)
    lower = completion["content"].lower().strip()

    def matched_choice(m: re.Match) -> DilemmaChoice:
        return choice_from_str(m.group(1), participant_condition)

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
    game_setup: GameSetup[DilemmaChoice, str] = GameSetup(
        num_rounds=NUM_ROUNDS,
        generate_instruction_prompt=get_prompt_pd,
        payoffs=payoffs_pd,
        extract_choice=extract_choice_pd,
        next_round=simultaneous.next_round,
        analyse_rounds=simultaneous.analyse_rounds,
        model_setup=model_setup,
    )
    measurement_setup: MeasurementSetup[DilemmaChoice] = MeasurementSetup(
        num_replications=sample_size,
        compute_freq=compute_freq_pd,
        choose_participant_condition=lambda: randomized(
            {
                CONDITION_CHAIN_OF_THOUGHT: [True, False],
                CONDITION_LABEL: [label.value for label in Label],
                CONDITION_CASE: [case.value for case in Case],
                CONDITION_PRONOUN: [pronoun.value for pronoun in Pronoun],
                CONDITION_DEFECT_FIRST: [True, False],
                CONDITION_LABELS_REVERSED: [True, False],
            }
        ),
    )
    return run_experiment(
        ai_participants=AI_PARTICIPANTS,
        partner_conditions={
            "unconditional cooperate": strategy_cooperate,
            "unconditional defect": strategy_defect,
            "tit for tat C": strategy_t4t_cooperate,
            "tit for tat D": strategy_t4t_defect,
        },
        measurement_setup=measurement_setup,
        game_setup=game_setup,
    )


if __name__ == "__main__":
    run_and_record_experiment("dilemma", run)
