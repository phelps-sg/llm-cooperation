import re
from enum import Enum
from typing import Dict, Hashable

from openai_pygenerator import Completion, content

from llm_cooperation import Choice, ModelSetup, amount_as_str
from llm_cooperation.experiments import AI_PARTICIPANTS, run_and_record_experiment
from llm_cooperation.gametypes.oneshot import OneShotResults, run_experiment

TOTAL_SHARE = 4

SAMPLE_SIZE = 30


class DictatorEnum(Enum):
    BLACK = 0
    BROWN = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


def project(color: str) -> str:
    return f"project {color}"


color_mappings: Dict[DictatorEnum, str] = {
    DictatorEnum.BLACK: "black",
    DictatorEnum.BROWN: "brown",
    DictatorEnum.GREEN: "green",
    DictatorEnum.BLUE: "blue",
    DictatorEnum.WHITE: "white",
}

reverse_color_mappings: Dict[str, DictatorEnum] = {
    value: key for key, value in color_mappings.items()
}


class DictatorChoice(Choice):
    @property
    def description(self) -> str:
        return project(color_mappings[self._value])

    def __init__(self, value: DictatorEnum):
        self._value: DictatorEnum = value

    @property
    def value(self) -> Hashable:
        return self._value

    @property
    def donation(self) -> float:
        return float(self._value.value)

    @property
    def payoff_ego(self) -> float:
        return TOTAL_SHARE - self.donation

    @property
    def payoff_allo(self) -> float:
        return self.donation


BLACK = DictatorChoice(DictatorEnum.BLACK)
BROWN = DictatorChoice(DictatorEnum.BROWN)
GREEN = DictatorChoice(DictatorEnum.GREEN)
BLUE = DictatorChoice(DictatorEnum.BLUE)
WHITE = DictatorChoice(DictatorEnum.WHITE)

all_dictator_choices = [BLACK, BROWN, GREEN, BLUE, WHITE]


def payout_ego(choice: DictatorChoice) -> str:
    return amount_as_str(choice.payoff_ego)


def payout_allo(choice: DictatorChoice) -> str:
    return amount_as_str(choice.payoff_allo)


def describe_payoffs(choice: DictatorChoice) -> str:
    return f"""
   If you choose '{choice.description}, then you will earn {payout_ego(choice)}
and your partner will also earn {payout_allo(choice)}.
    """


def choice_menu() -> str:
    return " | ".join([f"'{c.description}'" for c in all_dictator_choices])


def get_prompt_dictator(role_prompt: str) -> str:
    return f"""
{role_prompt}
This is a study of investment choices in different situations.
You will be making your choices with an investment
partner who is another person. She is in the next room waiting. In a minute I will take
you there to meet your investment partner. You and your partner will be involved in
the project. However, you are the sole decision maker. Your partner must accept
your decision. She is unable to make the investment decisions. The object is to earn
money for yourself.
{describe_payoffs(BLACK)}
{describe_payoffs(BROWN)}
{describe_payoffs(GREEN)}
{describe_payoffs(BLUE)}
{describe_payoffs(WHITE)}
You will make decisions that are consistent with the role outlined earlier,
but you are not allowed to communicate with your partner apart from informing
them of your choice. Your decision will be
final and cannot be changed after you have made it. Here is your investment
partner. Give your response in the format below:

Explanation: [why you picked your choice]
Choice: [{choice_menu()}]
"""


def extract_choice_dictator(completion: Completion) -> DictatorChoice:
    text = content(completion).lower()
    match = re.search(r"choice:\s*(.*)", text)
    if match:
        choice = match.group(1)
        for key, value in reverse_color_mappings.items():
            if key in choice:
                return DictatorChoice(value)
    raise ValueError(f"Cannot determine choice from {completion}")


def payoffs_dictator(player1: DictatorChoice) -> float:
    return player1.payoff_ego


def compute_freq_dictator(history: DictatorChoice) -> float:
    return history.donation / TOTAL_SHARE


def run_experiment_dictator(
    model_setup: ModelSetup, sample_size: int = SAMPLE_SIZE
) -> OneShotResults[DictatorChoice]:
    return run_experiment(
        ai_participants=AI_PARTICIPANTS,
        num_samples=sample_size,
        generate_instruction_prompt=get_prompt_dictator,
        extract_choice=extract_choice_dictator,
        payoffs=payoffs_dictator,
        compute_freq=compute_freq_dictator,
        model_setup=model_setup,
    )


if __name__ == "__main__":
    run_and_record_experiment(name="dictator", experiment=run_experiment_dictator)
