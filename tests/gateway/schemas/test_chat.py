import pytest
import pydantic

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

    with pytest.raises(pydantic.ValidationError, match="at least 1 items"):
        chat.RequestPayload(**{"messages": []})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        chat.RequestPayload(**{})


def test_chat_response():
    chat.ResponsePayload(
        **{
            "candidates": [
                {"message": {"role": "user", "content": "content"}, "metadata": {}},
            ],
            "metadata": {},
        }
    )
    chat.ResponsePayload(
        **{
            "candidates": [
                {
                    "message": {"role": "user", "content": "content"},
                    "metadata": {"i": 0, "f": 0.1, "s": "s", "b": True},
                },
            ],
            "metadata": {"i": 0, "f": 0.1, "s": "s", "b": True},
        }
    )

    with pytest.raises(pydantic.ValidationError, match="field required"):
        chat.ResponsePayload(**{"metadata": {}})
