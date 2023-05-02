import logging
import os
import time
from typing import Dict, Iterable

import openai
from openai.error import APIError, RateLimitError


def var(name: str, default: str) -> str:
    return os.environ.get(name) or default


GPT_ENGINE = var("GPT_ENGINE", "gpt-3.5-turbo")
GPT_TEMPERATURE = float(var("GPT_TEMPERATURE", "0.2"))
GPT_MAX_TOKENS = int(var("GPT_MAX_TOKENS", "100"))
GPT_MAX_RETRIES = int(var("GPT_MAX_RETRIES", "5"))
GPT_RETRY_EXPONENT_SECONDS = int(var("GPT_RETRY_EXPONENT_SECONDS", "2"))
GPT_RETRY_BASE_SECONDS = int(var("GPT_RETRY_BASE_SECONDS", "20"))

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

Completion = Dict[str, str]
Conversation = Iterable[Completion]


def generate_completions(
    messages: Conversation, n: int = 1, retries: int = 0
) -> Conversation:
    logger.debug("messages = %s", messages)
    try:
        result = openai.ChatCompletion.create(
            model=GPT_ENGINE,
            messages=messages,
            max_tokens=GPT_MAX_TOKENS,
            n=n,
            temperature=GPT_TEMPERATURE,
        )
        logger.debug("response = %s", result)
        for choice in result.choices:
            yield choice.message
    except (RateLimitError, APIError) as err:
        if isinstance(err, APIError) and not (err.http_status in [524, 502]):
            raise
        logger.warning("Error returned from openai API: %s", err)
        logger.debug("retries = %d", retries)
        if retries < GPT_MAX_RETRIES:
            logger.info("Retrying... ")
            time.sleep(GPT_RETRY_BASE_SECONDS + GPT_RETRY_EXPONENT_SECONDS**retries)
            for completion in generate_completions(messages, n, retries + 1):
                yield completion
        else:
            logger.error("Maximum retries reached, aborting.")
            raise


def user_message(content: str) -> Completion:
    return {"role": "user", "content": content}
