from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, transcript, user_message

from llm_cooperation import CT, Group, ModelSetup, Results, completer_for, logger

PromptGenerator = Callable[[str], str]
ResultSingleShotGame = Tuple[
    Group, str, float, float, Optional[CT], List[str], str, float
]


class OneShotResults(Results, Generic[CT]):
    def __init__(self, rows: Iterable[ResultSingleShotGame[CT]]):
        self._rows: Iterable[ResultSingleShotGame[CT]] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (str(group), prompt, score, freq, choices, history, model, temp)
                for group, prompt, score, freq, choices, history, model, temp in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Score",
                "Cooperation frequency",
                "Choice",
                "Transcript",
                "Model",
                "Temperature",
            ],
        )


def play_game(
    role_prompt: str,
    generate_instruction_prompt: PromptGenerator,
    model_setup: ModelSetup,
) -> List[Completion]:
    gpt_completions = completer_for(model_setup)
    messages = [user_message(generate_instruction_prompt(role_prompt))]
    messages += gpt_completions(messages, 1)
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Completion], CT],
) -> Tuple[float, CT]:
    ai_choice = extract_choice(conversation[1])
    logger.debug("ai_choice = %s", ai_choice)
    score = payoffs(ai_choice)
    return score, ai_choice


def analyse(
    conversation: List[Completion],
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Completion], CT],
    compute_freq: Callable[[CT], float],
) -> Tuple[float, float, Optional[CT], List[str]]:
    try:
        history = transcript(conversation)
        score, ai_choice = compute_scores(list(conversation), payoffs, extract_choice)
        freq = compute_freq(ai_choice)
        return score, freq, ai_choice, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_samples(
    prompt: str,
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Completion], CT],
    compute_freq: Callable[[CT], float],
    model_setup: ModelSetup,
) -> Iterable[Tuple[float, float, Optional[CT], List[str]]]:
    # pylint: disable=R0801
    for _i in range(num_samples):
        conversation = play_game(
            role_prompt=prompt,
            generate_instruction_prompt=generate_instruction_prompt,
            model_setup=model_setup,
        )
        yield analyse(conversation, payoffs, extract_choice, compute_freq)


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Completion], CT],
    compute_freq: Callable[[CT], float],
    model_setup: ModelSetup,
) -> OneShotResults[CT]:
    return OneShotResults(
        (
            group,
            prompt,
            score,
            freq,
            choices,
            history,
            model_setup.model,
            model_setup.temperature,
        )
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for score, freq, choices, history in generate_samples(
            prompt=prompt,
            num_samples=num_samples,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
            model_setup=model_setup,
        )
    )
