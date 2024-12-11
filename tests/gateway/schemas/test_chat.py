import pydantic
import pytest

from mlflow.gateway.schemas import chat


def test_chat_request():
    chat.RequestPayload(
        **{
            "messages": [{"role": "user", "content": "content"}],
        }
    )
    chat.RequestPayload(
        **{
            "messages": [{"role": "user", "content": "content"}],
            "n": 1000,
            "extra": "extra",
            "temperature": 2.0,
        }
    )

    with pytest.raises(pydantic.ValidationError, match="less than or equal to 2"):
        chat.RequestPayload(
            **{
                "messages": [{"role": "user", "content": "content"}],
                "temperature": 3.0,
            }
        )

    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        chat.RequestPayload(
            **{
                "messages": [{"role": "user", "content": "content"}],
                "stop": [],
            }
        )

    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        chat.RequestPayload(**{"messages": []})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.RequestPayload(**{})


def test_chat_response():
    chat.ResponsePayload(
        **{
            "created": 100,
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"role": "user", "content": "content"},
                    "index": 0,
                },
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 1,
            },
        }
    )

    chat.ResponsePayload(
        **{
            "id": "foobar",
            "created": 100,
            "model": "gpt-4",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "user", "content": "content"},
                    "finish_reason": "stop",
                    "index": 0,
                },
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 1,
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.ResponsePayload(**{"usage": {}})
