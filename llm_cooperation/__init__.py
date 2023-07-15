from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Hashable, Tuple, TypeVar

import openai_pygenerator
import pandas as pd
from openai_pygenerator import Completer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Choice(ABC):
    @property
    @abstractmethod
    def value(self) -> Hashable:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def __eq__(self, o: object) -> bool:
        if issubclass(type(o), Choice):
            return self.value.__eq__(o.value)  # type: ignore
        return False

    def __hash__(self) -> int:
        return self.value.__hash__()


Group = Enum(
    "Group",
    ["Cooperative", "Competitive", "Altruistic", "Selfish", "Mixed", "Control"],
)


class Results(ABC):
    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        pass


@dataclass
class ModelSetup:
    model: str
    temperature: float
    max_tokens: int


Experiment = Callable[[ModelSetup, int], Results]

DEFAULT_MODEL_SETUP = ModelSetup(
    model=openai_pygenerator.GPT_MODEL,
    temperature=openai_pygenerator.GPT_TEMPERATURE,
    max_tokens=openai_pygenerator.GPT_MAX_TOKENS,
)

CT = TypeVar("CT", bound=Choice)
CT_co = TypeVar("CT_co", bound=Choice, covariant=True)
CT_contra = TypeVar("CT_contra", bound=Choice, contravariant=True)
PT = TypeVar("PT")
PT_contra = TypeVar("PT_contra", contravariant=True)

Payoffs = Tuple[float, float]
ChainOfThoughtCondition = bool


def amount_as_str(amount: float) -> str:
    return f"${amount:.02f}"


def completer_for(model_setup: ModelSetup) -> Completer:
    return openai_pygenerator.completer(
        model=model_setup.model,
        temperature=model_setup.temperature,
        max_tokens=model_setup.max_tokens,
    )
