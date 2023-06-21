import logging
import re
from abc import ABC
from enum import Enum, auto
from typing import Hashable, List, Optional

import numpy as np
from openai_pygenerator import Completion, content, user_message

from llm_cooperation import Choice, Payoffs, amount_as_str
from llm_cooperation.experiments import AI_PARTICIPANTS, run_and_record_experiment
from llm_cooperation.gametypes import alternating
from llm_cooperation.gametypes.repeated import (
    Choices,
    GameSetup,
    GameState,
    RepeatedGameResults,
    Strategy,
    run_experiment,
)

logger = logging.getLogger(__name__)

MAX_AMOUNT: float = 10.0
MIN_AMOUNT: float = 0.0

# NUM_ROUNDS = 22
# SAMPLE_SIZE = 30

NUM_ROUNDS: int = 6
SAMPLE_SIZE: int = 3


class ResponderEnum(Enum):
    Accept = auto()
    Reject = auto()


class UltimatumChoice(Choice, ABC):
    pass


class ProposerChoice(UltimatumChoice):
    def __init__(self, value: float):
        self._value: float = value

    @property
    def value(self) -> Hashable:
        return self._value

    @property
    def amount(self) -> float:
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


def next_round_ultimatum(
    partner_strategy: Strategy, state: GameState
) -> List[Completion]:
    user_choice = partner_strategy(state).description
    ai_to_propose = isinstance(user_choice, ResponderChoice)
    return [
        user_message(
            f"Your partner chose {user_choice} in that round. "
            f"""Now we will move on to the next round.
What is your {"proposal" if ai_to_propose else "offer"} for the next round?"""
        )
    ]


def strategy_cooperate(state: GameState) -> UltimatumChoice:
    _scores, prev_choices = state.last_round
    if isinstance(prev_choices.ai, ResponderChoice):
        return ProposerChoice(MAX_AMOUNT)
    elif isinstance(prev_choices.ai, ProposerChoice):
        return Accept
    else:
        raise ValueError(f"Unknown choice type: {prev_choices.ai}")


def get_prompt_ultimatum(num_rounds: int, role_prompt: str) -> str:
    return f"""
    {role_prompt}
This is a study of resource allocation choices. You will be making your decisions
with another participant who is also taking part in this study.
They are ready and waiting. In a moment I will bring you into the virtual room to meet them.
You and your partner will participate in a total of {num_rounds} rounds of resource allocation decisions.
The objective is to earn as much virtual currency as possible in total across all rounds.
In each round you will play one of two roles: proposer or responder.

When you are the proposer, you get {amount_as_str(MAX_AMOUNT)} to share. You can choose to offer the responder
any amount from {amount_as_str(MIN_AMOUNT)} to {amount_as_str(MAX_AMOUNT)} for the responder to keep.
The proposer would keep the rest. This is based on the response of the responder.

When you are the responder, as responder you can either accept or reject the offer. If the responder accepts the
offer then they keep the offer and the proposer gets {amount_as_str(MAX_AMOUNT)} less
the offered amount.

If the responder rejects the offer neither role gets any money. For example, if the proposer offered the
responder {amount_as_str(3)} and the responder accepted the offer, then the responder gets {amount_as_str(3)}
and the proposer gets {amount_as_str(7)}.
Therefore, the payoffs depend on your role. If the proposer offered the responder {amount_as_str(3)} and the
responder rejected the offer, then the proposer gets {amount_as_str(0)} and the responder gets {amount_as_str(0)}.
Both roles get {amount_as_str(0)} if the responder rejects the offer.
In every decision, you will act consistently with the role outlined earlier. Please note that there will be no
communication between you and your partner apart from the proposal or the decision to accept or reject.

When you are the proposer you will only state the dollar amount you will offer.
Response: [{amount_as_str(0)} to {amount_as_str(10)}]

When you are the responder, you only give
Response: [accept/reject]

In the first round you will be the proposer and I will be the responder and then we will switch.

Remember, {role_prompt}. What is your proposal?
    """


def amount_from_str(s: str) -> float:
    result = dollar_float_pattern.search(s)
    if result:
        return float(result.group(1))
    else:
        raise ValueError(f"Cannot extract dollar amount from {s}")


def extract_responder_choice(completion: Completion) -> Optional[ResponderChoice]:
    message = content(completion).lower().strip()
    if "accept" in message:
        return Accept
    elif "reject" in message:
        return Reject
    return None


def extract_proposer_choice(completion: Completion) -> ProposerChoice:
    return ProposerChoice(amount_from_str(content(completion)))


def extract_choice_ultimatum(completion: Completion, **kwargs: bool) -> UltimatumChoice:
    proposer = kwargs["proposer"]
    return (
        extract_proposer_choice(completion)
        if proposer
        else extract_responder_choice(completion)
    )


def compute_freq_ultimatum(choices: List[Choices]) -> float:
    return float(
        np.mean([c.amount for c in choices if isinstance(c, ProposerChoice)])
        / MAX_AMOUNT
    )


def _payoffs(proposer: ProposerChoice, responder: ResponderChoice) -> Payoffs:
    if responder == Reject:
        return 0.0, 0.0
    else:
        offered: float = proposer.amount
        remainder: float = MAX_AMOUNT - offered
        return remainder, offered


def payoffs_ultimatum(player1: UltimatumChoice, player2: UltimatumChoice) -> Payoffs:
    if isinstance(player1, ProposerChoice) and isinstance(player2, ResponderChoice):
        return _payoffs(player1, player2)
    elif isinstance(player1, ResponderChoice) and isinstance(player2, ProposerChoice):
        payoffs = _payoffs(player2, player1)
        return payoffs[1], payoffs[0]
    else:
        raise ValueError(f"Invalid choice combination: {player1}, {player2}")


def run_experiment_ultimatum() -> RepeatedGameResults:
    return run_experiment(
        ai_participants=AI_PARTICIPANTS,
        num_samples=SAMPLE_SIZE,
        partner_conditions={"cooperate": strategy_cooperate},
        compute_freq=compute_freq_ultimatum,
        game_setup=GameSetup(
            num_rounds=NUM_ROUNDS,
            generate_instruction_prompt=get_prompt_ultimatum,
            extract_choice=extract_choice_ultimatum,
            payoffs=payoffs_ultimatum,
            next_round=next_round_ultimatum,
            rounds=alternating.rounds_setup,
        ),
    )


if __name__ == "__main__":
    run_and_record_experiment(name="ultimatum", run=run_experiment_ultimatum)
