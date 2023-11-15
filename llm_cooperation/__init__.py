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

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
#

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import numpy as np
import openai_pygenerator
import pandas as pd
from openai_pygenerator import Completer, Completion, Completions, History
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

    def description(self, participant_condition: Settings) -> str:
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
    dry_run: Optional[str]


Experiment = Callable[[ModelSetup, int], Results]

DEFAULT_MODEL_SETUP = ModelSetup(
    model=openai_pygenerator.GPT_MODEL,
    temperature=openai_pygenerator.GPT_TEMPERATURE,
    max_tokens=openai_pygenerator.GPT_MAX_TOKENS,
    dry_run=None,
)

CT = TypeVar("CT", bound=Choice)
CT_co = TypeVar("CT_co", bound=Choice, covariant=True)
CT_contra = TypeVar("CT_contra", bound=Choice, contravariant=True)

RT = TypeVar("RT")
RT_contra = TypeVar("RT_contra", contravariant=True)

Payoffs = Tuple[float, float]


def assistant_message(description: str) -> Completion:
    return {"role": "assistant", "content": description}


def all_combinations(grid: Grid) -> itertools.product:
    return itertools.product(*grid.values())


def settings_from_combinations(
    keys: List[str], combinations: Iterable[ConfigValue]
) -> Settings:
    settings: Settings = dict()
    for i, value in enumerate(combinations):
        settings[keys[i]] = value
    return settings


def randomized(grid: Grid) -> Settings:
    keys = list(grid.keys())
    combinations = list(all_combinations(grid))
    num_combinations = len(combinations)
    random_index: int = int(np.random.randint(num_combinations))
    return settings_from_combinations(keys, combinations[random_index])


def exhaustive(grid: Grid) -> Iterable[Settings]:
    variables = list(grid.keys())
    for values in all_combinations(grid):
        yield settings_from_combinations(variables, values)


def amount_as_str(amount: float) -> str:
    return f"${amount:.02f}"


def completer_for(model_setup: ModelSetup) -> Completer:
    if model_setup.dry_run is not None:
        dummy_completions: Tuple[Completion] = (assistant_message(model_setup.dry_run),)

        def dummy_completer(__history__: History, __n__: int) -> Completions:
            return iter(dummy_completions)

        return dummy_completer
    return openai_pygenerator.completer(
        model=model_setup.model,
        temperature=model_setup.temperature,
        max_tokens=model_setup.max_tokens,
    )
