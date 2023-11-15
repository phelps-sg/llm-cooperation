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

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, transcript

from llm_cooperation import CT, RT, Group, ModelSetup, Results, Settings, logger
from llm_cooperation.gametypes import PromptGenerator, start_game

ResultSingleShotGame = Tuple[
    Group,
    RT,
    Settings,
    float,
    float,
    Optional[CT],
    List[str],
    str,
    float,
]


class OneShotResults(Results, Generic[CT, RT]):
    def __init__(self, rows: Iterable[ResultSingleShotGame[RT, CT]]):
        self._rows: Iterable[ResultSingleShotGame[RT, CT]] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    str(group),
                    prompt,
                    condition,
                    score,
                    freq,
                    choice,
                    history,
                    model,
                    temp,
                )
                # pylint: disable=line-too-long
                for group, prompt, condition, score, freq, choice, history, model, temp in self._rows
            ],
            columns=[
                "Group",
                "Participant",
                "Condition",
                "Score",
                "Cooperation frequency",
                "Choice",
                "Transcript",
                "Model",
                "Temperature",
            ],
        )


def play_game(
    role_prompt: RT,
    participant_condition: Settings,
    generate_instruction_prompt: PromptGenerator[RT],
    model_setup: ModelSetup,
) -> List[Completion]:
    gpt_completions, messages = start_game(
        generate_instruction_prompt, model_setup, participant_condition, role_prompt
    )
    # gpt_completions = completer_for(model_setup)
    # messages = [user_message(generate_instruction_prompt(role_prompt))]
    messages += gpt_completions(messages, 1)
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Settings, Completion], CT],
    participant_condition: Settings,
) -> Tuple[float, CT]:
    ai_choice = extract_choice(participant_condition, conversation[1])
    logger.debug("ai_choice = %s", ai_choice)
    score = payoffs(ai_choice)
    return score, ai_choice


def analyse(
    conversation: List[Completion],
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Settings, Completion], CT],
    compute_freq: Callable[[CT], float],
    participant_condition: Settings,
) -> Tuple[float, float, Optional[CT], List[str], Settings]:
    try:
        history = transcript(conversation)
        score, ai_choice = compute_scores(
            list(conversation), payoffs, extract_choice, participant_condition
        )
        freq = compute_freq(ai_choice)
        return score, freq, ai_choice, history, participant_condition
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)], dict()


def generate_replications(
    prompt: RT,
    num_replications: int,
    generate_instruction_prompt: PromptGenerator[RT],
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Settings, Completion], CT],
    compute_freq: Callable[[CT], float],
    model_setup: ModelSetup,
    choose_participant_condition: Callable[[], Settings],
) -> Iterable[Tuple[float, float, Optional[CT], List[str], Settings]]:
    # pylint: disable=R0801
    for __i__ in range(num_replications):
        condition = choose_participant_condition()
        conversation = play_game(
            role_prompt=prompt,
            participant_condition=condition,
            generate_instruction_prompt=generate_instruction_prompt,
            model_setup=model_setup,
        )
        yield analyse(conversation, payoffs, extract_choice, compute_freq, condition)


def run_experiment(
    ai_participants: Dict[Group, List[RT]],
    num_replications: int,
    generate_instruction_prompt: PromptGenerator[RT],
    payoffs: Callable[[CT], float],
    extract_choice: Callable[[Settings, Completion], CT],
    compute_freq: Callable[[CT], float],
    model_setup: ModelSetup,
    choose_participant_condition: Callable[[], Settings],
) -> OneShotResults[CT, RT]:
    return OneShotResults(
        (
            group,
            participant,
            participant_condition,
            score,
            freq,
            choices,
            history,
            model_setup.model,
            model_setup.temperature,
        )
        for group, participants in ai_participants.items()
        for participant in participants
        for score, freq, choices, history, participant_condition in generate_replications(
            prompt=participant,
            num_replications=num_replications,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
            model_setup=model_setup,
            choose_participant_condition=choose_participant_condition,
        )
    )
