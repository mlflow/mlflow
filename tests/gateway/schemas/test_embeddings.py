import pydantic
import pytest

from mlflow.gateway.schemas import embeddings


def test_embeddings_request():
    embeddings.RequestPayload(**{"text": "text"})
    embeddings.RequestPayload(**{"text": ""})
    embeddings.RequestPayload(**{"text": ["prompt"]})
    embeddings.RequestPayload(**{"text": "text", "extra": "extra", "another_extra": 1})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        embeddings.RequestPayload(**{})


def test_embeddings_response():
    embeddings.ResponsePayload(
        **{
            "embeddings": [[0.1, 0.2, 0.3]],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 1,
                "model": "gpt-4",
                "route_type": "llm/v1/embeddings",
            },
        }
    )
    embeddings.ResponsePayload(
        **{
            "embeddings": [[0.1, 0.2, 0.3]],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 1,
                "model": "gpt-4",
                "route_type": "llm/v1/embeddings",
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        embeddings.ResponsePayload(**{"metadata": {}})
