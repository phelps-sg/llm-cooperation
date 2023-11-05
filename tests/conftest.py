from typing import List

import pytest
from openai_pygenerator import Completion


@pytest.fixture
def conversation() -> List[Completion]:
    return [
        {"user": "scenario prompt.  What is your choice in the first round?"},
        {"role": "assistant", "content": "project green"},
        {"role": "user", "content": "Your partner chose project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "Your partner chose project green"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "Your partner chose project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "Your partner chose project blue"},
        {"role": "assistant", "content": "project green"},
        {"role": "user", "content": "project green"},
    ]
