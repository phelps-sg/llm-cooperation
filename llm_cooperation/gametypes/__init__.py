from __future__ import annotations

from typing import List, Protocol, Tuple

from openai_pygenerator import Completer, Completion, user_message

from llm_cooperation import PT, RT, ModelSetup, PT_contra, RT_contra, completer_for


class PromptGenerator(Protocol[PT_contra, RT_contra]):
    def __call__(self, condition: PT_contra, role_prompt: RT_contra) -> str:
        ...


def start_game(
    prompt_generator: PromptGenerator[PT, RT],
    model_setup: ModelSetup,
    participant_condition: PT,
    role_prompt: RT,
) -> Tuple[Completer, List[Completion]]:
    gpt_completions = completer_for(model_setup)
    messages: List[Completion] = [
        user_message(prompt_generator(participant_condition, role_prompt))
    ]
    return gpt_completions, messages
