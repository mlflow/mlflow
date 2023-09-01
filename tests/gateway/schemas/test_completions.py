import pydantic
import pytest

from mlflow.gateway.schemas import completions


def test_completions_request():
    completions.RequestPayload(**{"prompt": "prompt"})
    completions.RequestPayload(**{"prompt": ""})
    completions.RequestPayload(**{"prompt": "", "extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        completions.RequestPayload(**{"extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        completions.RequestPayload(**{})


def test_completions_response():
    completions.ResponsePayload(
        **{
            "candidates": [{"text": "text", "metadata": {}}],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 1,
                "model": "gpt-4",
                "route_type": "llm/v1/completions",
            },
        }
    )
    completions.ResponsePayload(
        **{
            "candidates": [
                {
                    "text": "text",
                    "metadata": {"finish_reason": "length"},
                }
            ],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 1,
                "model": "gpt-4",
                "route_type": "llm/v1/completions",
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        completions.ResponsePayload(**{"metadata": {}})
