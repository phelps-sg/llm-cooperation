from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Hashable, Iterable, List, Protocol, Tuple, TypeVar

import numpy as np
import openai_pygenerator
import pandas as pd
from openai_pygenerator import Completer
from plotly.basedatatypes import itertools

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

ConfigValue = float | str | bool
Grid = Dict[str, List[ConfigValue]]
Settings = Dict[str, ConfigValue]


class Choice(Protocol):
    @property
    def value(self) -> Hashable:
        ...

    @property
    def description(self) -> str:
        ...


Group = Enum(
    "Group",
    ["Cooperative", "Competitive", "Altruistic", "Selfish", "Mixed", "Control"],
)


class Results(ABC):
    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        pass


@dataclass(frozen=True)
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

RT = TypeVar("RT")
RT_contra = TypeVar("RT_contra", contravariant=True)

Payoffs = Tuple[float, float]


def all_combinations(grid: Grid) -> itertools.product:
    return itertools.product(*grid.values())


def settings_from_combinations(
    keys: List[str], combinations: Iterable[ConfigValue]
) -> Settings:
    settings: Settings = dict()
    for i, value in enumerate(combinations):
        settings[keys[i]] = value
    return settings


def randomized(n: int, grid: Grid) -> Iterable[Settings]:
    keys = list(grid.keys())
    combinations = list(all_combinations(grid))
    num_combinations = len(combinations)
    for __i__ in range(n):
        random_index: int = int(np.random.randint(num_combinations))
        yield settings_from_combinations(keys, combinations[random_index])


def exhaustive(grid: Grid) -> Iterable[Settings]:
    keys = list(grid.keys())
    for values in all_combinations(grid):
        yield settings_from_combinations(keys, values)


def amount_as_str(amount: float) -> str:
    return f"${amount:.02f}"


def completer_for(model_setup: ModelSetup) -> Completer:
    return openai_pygenerator.completer(
        model=model_setup.model,
        temperature=model_setup.temperature,
        max_tokens=model_setup.max_tokens,
    )
