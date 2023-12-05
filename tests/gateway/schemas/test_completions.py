import time

import pydantic
import pytest

from mlflow.gateway.schemas import completions


def test_completions_request():
    completions.RequestPayload(**{"prompt": "prompt"})
    completions.RequestPayload(**{"prompt": ""})
    completions.RequestPayload(**{"prompt": "", "extra": "extra", "temperature": 2.0})

    with pytest.raises(pydantic.ValidationError, match="less than or equal to 2"):
        completions.RequestPayload(
            **{
                "messages": [{"role": "user", "content": "content"}],
                "temperature": 3.0,
            }
        )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        completions.RequestPayload(**{"extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        completions.RequestPayload(**{})


def test_completions_response():
    completions.ResponsePayload(
        id="completions-id-1",
        object="text_completion",
        created=int(time.time()),
        model="gpt-4",
        choices=[
            completions.Choice(
                index=0,
                text="text",
                finish_reason="stop",
            )
        ],
        usage=completions.CompletionsUsage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )

    completions.ResponsePayload(
        object="text_completion",
        created=int(time.time()),
        model="hf-tgi",
        choices=[
            completions.Choice(
                index=0,
                text="text",
                finish_reason="stop",
            ),
            completions.Choice(
                index=1,
                text="text",
                finish_reason="length",
            ),
            completions.Choice(
                index=2,
                text="text",
                finish_reason="foo",
            ),
            completions.Choice(
                index=3,
                text="text",
                finish_reason=None,
            ),
        ],
        usage=completions.CompletionsUsage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        ),
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        completions.ResponsePayload(**{"usage": {}})
