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
            "extra": "extra",
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
            "candidates": [
                {
                    "message": {"role": "user", "content": "content"},
                    "metadata": {},
                },
            ],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 1,
                "model": "gpt-4",
                "route_type": "llm/v1/chat",
            },
        }
    )
    chat.ResponsePayload(
        **{
            "candidates": [
                {
                    "message": {"role": "user", "content": "content"},
                    "metadata": {"finish_reason": "stop"},
                },
            ],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 1,
                "model": "gpt-4",
                "route_type": "llm/v1/chat",
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.ResponsePayload(**{"metadata": {}})
