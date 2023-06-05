import re
from abc import ABC
from enum import Enum, auto
from typing import Hashable, Iterable, List, Tuple

from openai_pygenerator import Completion

from llm_cooperation import (
    AI_PARTICIPANTS,
    Choice,
    Choices,
    ResultRow,
    Scores,
    run_experiment,
)

MAX_AMOUNT: float = 10.0
MIN_AMOUNT: float = 0.0

NUM_ROUNDS = 22
SAMPLE_SIZE = 30


def amount_as_str(amount: float) -> str:
    return f"${amount:.02f}"


def get_prompt_ultimatum(num_rounds: int) -> str:
    return f"""This is a study of resource allocation choices. You will be making your decisions
with another participant who is also taking part in this study.
They are ready and waiting. In a moment I will bring you into the virtual room to meet them.
You and your partner will participate in a total of {num_rounds} rounds of resource allocation decisions.
The objective is to earn as much virtual currency as possible in total across all rounds.
In each round you will play one of two roles: proposer or responder.
When proposer: you get {amount_as_str(MAX_AMOUNT)} to share. You can choose to offer the responder
any amount from {amount_as_str(MIN_AMOUNT)} to {amount_as_str(MAX_AMOUNT)} for the responder to keep.
The proposer would keep the rest. This is based on the response of the responder.
When the responder: the responder can either accept or reject the offer. If the responder accepts the offer then they
keep the offer and the proposer gets {amount_as_str(MAX_AMOUNT)} less the offered amount.
If the responder rejects the offer neither role gets any money. For example, if the proposer offered the
responder {amount_as_str(3)} and the responder accepted the offer, then the responder gets {amount_as_str(3)} and the
proposer gets {amount_as_str(7)}.
Therefore, the payoffs depend on your role. If the proposer offered the responder {amount_as_str(3)} and the
responder rejected the offer, then the proposer gets {amount_as_str(0)} and the responder gets {amount_as_str(0)}.
Both roles get {amount_as_str(0)} if the responder rejects the offer.
In every decision, you will act consistently with the role outlined earlier. Please note that there will be no
communication between you and your partner apart from the proposal or the decision to accept or reject.

When you are the proposer you will only state the dollar amount you will offer.
Response: [{amount_as_str(0)} to {amount_as_str(10)}]

When you are the responder, you only give
 Response: [accept/reject]

In the first round I will be the proposer and you will be the responder and then we will switch.
"""


class ResponderEnum(Enum):
    Accept = auto()
    Reject = auto()


class UltimatumChoice(Choice, ABC):
    pass


class ProposerChoice(UltimatumChoice):
    def __init__(self, value: float):
        self._value = value

    @property
    def value(self) -> Hashable:
        return self._value

    @property
    def description(self) -> str:
        return amount_as_str(self._value)


class ResponderChoice(UltimatumChoice):
    def __init__(self, value: ResponderEnum):
        self._value = value

    @property
    def value(self) -> Hashable:
        return self._value

    @property
    def description(self) -> str:
        if self._value is ResponderEnum.Accept:
            return "Accept"
        elif self._value is ResponderEnum.Reject:
            return "Reject"
        else:
            raise ValueError(f"Invalid value: ${self._value}")


Accept = ResponderChoice(ResponderEnum.Accept)
Reject = ResponderChoice(ResponderEnum.Reject)


dollar_float_pattern = re.compile(r"\$(\d+(?:\.\d+)?)")


def amount_from_str(s: str) -> float:
    result = dollar_float_pattern.search(s)

    if result:
        return float(result.group(1))
    else:
        raise ValueError(f"Cannot extract dollar amount from {s}")


def extract_choice_ultimatum(completion: Completion) -> Choice:
    content = completion["content"].lower().strip()
    if "accept" in content:
        return Accept
    elif "reject" in content:
        return Reject
    else:
        return ProposerChoice(amount_from_str(content))


def compute_freq_ultimatum(_choices: List[Choices]) -> float:
    pass


def analyse_round_ultimatum(
    _i: int, _conversation: List[Completion]
) -> Tuple[Scores, Choices]:
    pass


def payoffs_ultimatum(_player1: Choice, _player2: Choice) -> Tuple[int, int]:
    pass


def run_experiment_ultimatum() -> Iterable[ResultRow]:
    return run_experiment(
        ai_participants=AI_PARTICIPANTS,
        user_conditions={},
        num_rounds=NUM_ROUNDS,
        num_samples=SAMPLE_SIZE,
        generate_instruction_prompt=get_prompt_ultimatum,
        extract_choice=extract_choice_ultimatum,
        payoffs=payoffs_ultimatum,
        compute_freq=compute_freq_ultimatum,
    )
