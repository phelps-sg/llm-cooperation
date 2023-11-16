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

from typing import List, Protocol, Tuple

from openai_pygenerator import Completer, Completion, logging, user_message

from llm_cooperation import ModelSetup, Participant, completer_for

logger = logging.getLogger(__name__)


class PromptGenerator(Protocol):
    def __call__(self, participant: Participant) -> str:
        ...


def start_game(
    prompt_generator: PromptGenerator,
    model_setup: ModelSetup,
    participant: Participant,
) -> Tuple[Completer, List[Completion]]:
    logger.debug("model_setup = %s", model_setup)
    gpt_completions = completer_for(model_setup)
    messages: List[Completion] = [user_message(prompt_generator(participant))]
    logger.debug("gpt_completions = %s", gpt_completions)
    return gpt_completions, messages
