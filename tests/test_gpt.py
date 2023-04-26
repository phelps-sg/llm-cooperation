from typing import Iterable

import pytest
from openai.error import RateLimitError, APIError
from openai.openai_object import OpenAIObject

from gpt import generate_completions, GPT_MAX_RETRIES


def aio(text: str) -> OpenAIObject:
    result = OpenAIObject()
    result["message"] = text
    return result


class MockChoices:
    def __init__(self, responses: Iterable[str]):
        self.choices = [aio(text) for text in responses]


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("openai.ChatCompletion.create")


@pytest.fixture
def mock_sleep(mocker):
    return mocker.patch("time.sleep", return_value=None)


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited", http_status=429),
        APIError("Gateway Timeout", http_status=524),
    ],
)
def test_generate_completion(mock_openai, mock_sleep, error):
    mock_openai.side_effect = [
        error,
        error,
        MockChoices(["Test completion 1", "Test completion 2"]),
    ]

    completions = list(generate_completions([]))

    assert completions == ["Test completion 1", "Test completion 2"]
    assert mock_sleep.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited", http_status=429),
        APIError("Gateway Timeout", http_status=524),
    ],
)
def test_generate_completion_error(mock_openai, mock_sleep, error):
    mock_openai.side_effect = [error] * GPT_MAX_RETRIES

    with pytest.raises(Exception):
        _ = list(generate_completions([]))

    assert mock_sleep.call_count == GPT_MAX_RETRIES - 1
