from typing import List

import pytest
from openai_pygenerator import Completion, user_message

from tests.test_ultimatum import assistant_message


@pytest.fixture
def conversation() -> List[Completion]:
    return [
        {"user": "scenario prompt"},
        {"role": "assistant", "content": "project green"},
        {"role": "user", "content": "project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project green"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project green"},
    ]


@pytest.fixture
def alternating_history() -> List[Completion]:
    return [
        assistant_message("Propose $10"),
        user_message("Accept, Propose $10"),
        assistant_message("Accept / Propose $10"),
        user_message("Accept / Propose $10"),
        assistant_message("I Accept, and then I propose $10"),
        user_message("Accept / Propose $10"),
    ]
