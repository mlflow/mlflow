import pytest
import pydantic

from mlflow.gateway.schemas import completions


def test_completions_request():
    completions.RequestPayload(**{"prompt": "prompt"})
    completions.RequestPayload(**{"prompt": ""})
    completions.RequestPayload(**{"prompt": "", "extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        completions.RequestPayload(**{"extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        completions.RequestPayload(**{})


def test_completions_response():
    completions.ResponsePayload(
        **{
            "candidates": [{"text": "text", "metadata": {}}],
            "metadata": {},
        }
    )
    completions.ResponsePayload(
        **{
            "candidates": [
                {
                    "text": "text",
                    "metadata": {"i": 0, "f": 0.1, "s": "s", "b": True},
                }
            ],
            "metadata": {"i": 0, "f": 0.1, "s": "s", "b": True},
        }
    )

    with pytest.raises(pydantic.ValidationError, match="field required"):
        completions.ResponsePayload(**{"metadata": {}})
