import pytest
import pydantic

from mlflow.gateway.schemas import chat


def test_chat_request():
    chat.Request(
        **{
            "messages": [{"role": "user", "content": "content"}],
        }
    )
    chat.Request(
        **{
            "messages": [{"role": "user", "content": "content"}],
            "extra": "extra",
        }
    )

    with pytest.raises(pydantic.ValidationError, match="at least 1 items"):
        chat.Request(**{"messages": []})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        chat.Request(**{})


def test_chat_response():
    chat.Response(
        **{
            "candidates": [{"role": "user", "content": "content"}],
        }
    )
    chat.Response(
        **{
            "candidates": [{"role": "user", "content": "content"}],
            "metadata": {
                "i": 0,
                "f": 0.1,
                "s": "s",
                "b": True,
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match="field required"):
        chat.Response(**{"metadata": {}})
