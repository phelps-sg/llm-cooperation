from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Hashable, Tuple, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)


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
    def to_df(self) -> pd.DataFrame:
        pass


CT = TypeVar("CT", bound=Choice)

Payoffs = Tuple[float, float]


def amount_as_str(amount: float) -> str:
    return f"${amount:.02f}"
