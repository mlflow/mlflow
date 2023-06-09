import pytest
import pydantic

from mlflow.gateway.schemas import embeddings


def test_embeddings_request():
    embeddings.Request(**{"text": "text"})
    embeddings.Request(**{"text": ""})
    embeddings.Request(**{"text": "text", "extra": "extra", "another_extra": 1})

    with pytest.raises(pydantic.ValidationError, match="field required"):
        embeddings.Request(**{})


def test_embeddings_response():
    embeddings.Response(
        **{
            "embeddings": [0.1, 0.2, 0.3],
        }
    )
    embeddings.Response(
        **{
            "embeddings": [0.1, 0.2, 0.3],
            "metadata": {
                "i": 0,
                "f": 0.1,
                "s": "s",
                "b": True,
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match="field required"):
        embeddings.Response(**{"metadata": {}})
