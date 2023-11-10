from __future__ import annotations

from typing import List, Protocol, Tuple

from openai_pygenerator import Completer, Completion, logging, user_message

from llm_cooperation import RT, ModelSetup, RT_contra, Settings, completer_for

logger = logging.getLogger(__name__)


class PromptGenerator(Protocol[RT_contra]):
    def __call__(self, condition: Settings, role_prompt: RT_contra) -> str:
        ...


def start_game(
    prompt_generator: PromptGenerator[RT],
    model_setup: ModelSetup,
    participant_condition: Settings,
    role_prompt: RT,
) -> Tuple[Completer, List[Completion]]:
    logger.debug("model_setup = %s", model_setup)
    gpt_completions = completer_for(model_setup)
    messages: List[Completion] = [
        user_message(prompt_generator(participant_condition, role_prompt))
    ]
    logger.debug("gpt_completions = %s", gpt_completions)
    return gpt_completions, messages
