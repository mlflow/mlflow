import pydantic
import pytest

from mlflow.gateway.schemas import embeddings


def test_embeddings_request():
    embeddings.RequestPayload(**{"input": "text"})
    embeddings.RequestPayload(**{"input": ""})
    embeddings.RequestPayload(**{"input": ["prompt"]})
    embeddings.RequestPayload(**{"input": "text", "extra": "extra", "another_extra": 1})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        embeddings.RequestPayload(**{})


def test_embeddings_response():
    embeddings.ResponsePayload(
        **{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.01, -0.1], "index": 0}],
            "model": "sentence-piece",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
    )

    embeddings.ResponsePayload(
        **{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.01, -0.1], "index": 0},
                {"object": "embedding", "embedding": [0.03, -0.03], "index": 1},
            ],
            "model": "sentence-piece",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        embeddings.ResponsePayload(**{"usage": {}})
