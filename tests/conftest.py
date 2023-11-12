from typing import List

import pytest
from openai_pygenerator import Completion

from llm_cooperation import ConfigValue, Settings
from llm_cooperation.experiments import CONDITION_CASE, Case
from llm_cooperation.experiments.dilemma import (
    CONDITION_CHAIN_OF_THOUGHT,
    CONDITION_DEFECT_FIRST,
    CONDITION_LABEL,
    CONDITION_LABELS_REVERSED,
    Label,
)

COLOR_COOPERATE = "Green"
COLOR_DEFECT = "Blue"


@pytest.fixture
def conversation() -> List[Completion]:
    return [
        {"user": "scenario prompt.  What is your choice in the first round?"},
        {"role": "assistant", "content": f"project {COLOR_COOPERATE}"},
        {"role": "user", "content": f"Your partner chose project {COLOR_DEFECT}"},
        {"role": "assistant", "content": f"project {COLOR_DEFECT}"},
        {"role": "user", "content": f"Your partner chose project {COLOR_COOPERATE}"},
        {"role": "assistant", "content": f"project {COLOR_DEFECT}"},
        {"role": "user", "content": f"Your partner chose project {COLOR_DEFECT}"},
        {"role": "assistant", "content": f"project {COLOR_DEFECT}"},
        {"role": "user", "content": f"Your partner chose project {COLOR_DEFECT}"},
        {"role": "assistant", "content": f"project {COLOR_COOPERATE}"},
        {"role": "user", "content": f"project {COLOR_COOPERATE}"},
    ]


@pytest.fixture
def base_condition() -> Settings:
    return {
        CONDITION_LABEL: Label.COLORS.value,
        CONDITION_LABELS_REVERSED: False,
        CONDITION_CHAIN_OF_THOUGHT: False,
        CONDITION_DEFECT_FIRST: False,
        CONDITION_CASE: Case.STANDARD.value,
    }


def modify_condition(
    base_condition: Settings, key: str, value: ConfigValue
) -> Settings:
    result = base_condition.copy()
    result[key] = value
    return result


@pytest.fixture
def with_chain_of_thought(base_condition: Settings) -> Settings:
    return modify_condition(base_condition, CONDITION_CHAIN_OF_THOUGHT, True)


@pytest.fixture
def with_numerals(base_condition: Settings) -> Settings:
    return modify_condition(base_condition, CONDITION_LABEL, Label.NUMERALS.value)


@pytest.fixture
def with_numbers(base_condition: Settings) -> Settings:
    return modify_condition(base_condition, CONDITION_LABEL, Label.NUMBERS.value)
