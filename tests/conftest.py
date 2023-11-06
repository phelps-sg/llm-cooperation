from typing import List

import pytest
from openai_pygenerator import Completion

from llm_cooperation.experiments.dilemma import COLOR_COOPERATE, COLOR_DEFECT


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
