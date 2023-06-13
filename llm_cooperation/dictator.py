from enum import Enum
from typing import Hashable

from llm_cooperation import Choice

TOTAL_SHARE = 4


class DictatorEnum(Enum):
    BLACK = 0
    BROWN = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


class DictatorChoice(Choice):
    @property
    def description(self) -> str:
        def project(color: str) -> str:
            return f"project {color}"

        if self._value is DictatorEnum.BLACK:
            return project("black")
        elif self._value is DictatorEnum.BROWN:
            return project("brown")
        elif self._value is DictatorEnum.GREEN:
            return project("green")
        elif self._value is DictatorEnum.BLUE:
            return project("blue")
        elif self._value is DictatorEnum.WHITE:
            return project("white")

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


black = DictatorChoice(DictatorEnum.BLACK)
brown = DictatorChoice(DictatorEnum.BROWN)
green = DictatorChoice(DictatorEnum.GREEN)
blue = DictatorChoice(DictatorEnum.BLUE)
white = DictatorChoice(DictatorEnum.WHITE)


def payout(amount: float) -> str:
    return f"${amount}.00"


def payout_ego(choice: DictatorChoice) -> str:
    return payout(choice.payoff_ego)


def payout_allo(choice: DictatorChoice) -> str:
    return payout(choice.payoff_allo)


def describe_payoffs(choice: DictatorChoice) -> str:
    return f"""
   If you choose '{choice.description}, then you will earn {payout_ego(choice)}
and your partner will also earn {payout_allo(choice)}.
    """


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
{describe_payoffs(black)}
{describe_payoffs(brown)}
{describe_payoffs(green)}
{describe_payoffs(blue)}
{describe_payoffs(white)}
You will make decisions that are consistent with the role outlined earlier,
but you are not allowed to communicate with your partner apart from informing
them of your choice. Your decision will be
final and cannot be changed after you have made it. Here is your investment
partner. What is your choice? Choice: [‘project black’ | ‘project brown’ | ‘project
green’ | ‘project blue’ | ‘project white’]
"""
