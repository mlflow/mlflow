import pytest
import pydantic

from mlflow.gateway.schemas import completions


def test_completions_request():
    completions.Request(**{"prompt": "prompt"})
    completions.Request(**{"prompt": ""})
    completions.Request(**{"prompt": "", "extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        completions.Request(**{"extra": "extra"})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        completions.Request(**{})


def test_completions_response():
    completions.Response(**{"candidates": [{"role": "user", "content": "content"}]})
    completions.Response(
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
        completions.Response(**{"metadata": {}})
