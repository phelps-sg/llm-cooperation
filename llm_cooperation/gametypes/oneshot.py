from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, gpt_completions, transcript, user_message

from llm_cooperation import CT, Group, Results, logger

PromptGenerator = Callable[[str], str]
ResultSingleShotGame = Tuple[Group, str, float, float, Optional[CT], List[str]]


class OneShotResults(Results):
    def __init__(self, rows: Iterable[ResultSingleShotGame]):
        self._rows: Iterable[ResultSingleShotGame] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (str(group), prompt, score, freq, choices, history)
                for group, prompt, score, freq, choices, history in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Score",
                "Cooperation frequency",
                "Choice",
                "Transcript",
            ],
        )


def play_game(
    role_prompt: str, generate_instruction_prompt: PromptGenerator
) -> List[Completion]:
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
) -> Iterable[Tuple[float, float, Optional[CT], List[str]]]:
    # pylint: disable=R0801
    for _i in range(num_samples):
        conversation = play_game(
            role_prompt=prompt,
            generate_instruction_prompt=generate_instruction_prompt,
        )
        yield analyse(conversation, payoffs, extract_choice, compute_freq)


def run_experiment(
    ai_participants: Dict[Group, List[str]],
    num_samples: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Completion], CT],
    compute_freq: Callable[[CT], float],
) -> OneShotResults:
    return OneShotResults(
        (group, prompt, score, freq, choices, history)
        for group, prompts in ai_participants.items()
        for prompt in prompts
        for score, freq, choices, history in generate_samples(
            prompt=prompt,
            num_samples=num_samples,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
        )
    )