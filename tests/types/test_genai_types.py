import pytest
from pydantic import ValidationError

from mlflow.types.chat import ChatCompletionResponse


def test_instantiation_chat_completion():
    response_structure = {
        "id": "1",
        "object": "1",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "user", "content": "hi"},
                "finish_reason": None,
            },
            {
                "index": 1,
                "message": {"role": "user", "content": "there"},
                "finish_reason": "STOP",
            },
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 22, "total_tokens": 34},
    }

    response = ChatCompletionResponse(**response_structure)

    assert response.id == "1"
    assert response.object == "1"
    assert response.created == 1
    assert response.model == "model"
    assert len(response.choices) == 2
    assert response.choices[0].index == 0
    assert response.choices[0].message.content == "hi"
    assert response.choices[1].finish_reason == "STOP"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 22
    assert response.usage.total_tokens == 34


def test_invalid_chat_completion():
    invalid_response_structure = {
        "id": "1",
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "user", "content": "hi"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 22, "total_tokens": 34},
    }

    with pytest.raises(ValidationError, match="1 validation error for ChatCompletionResponse"):
        ChatCompletionResponse(**invalid_response_structure)
