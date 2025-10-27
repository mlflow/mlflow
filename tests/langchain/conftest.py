import importlib

import openai
import pytest
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel

from tests.helper_functions import start_mock_openai_server
from tests.tracing.helper import reset_autolog_state  # noqa: F401


@pytest.fixture(autouse=True)
def set_envs(monkeypatch, mock_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai)
    monkeypatch.setenv("SERPAPI_API_KEY", "test")
    importlib.reload(openai)


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url


@pytest.fixture(autouse=True)
def reset_autolog(reset_autolog_state):
    # Apply the reset_autolog_state fixture to all tests for LangChain
    return


# Define a special embedding for testing
class DeterministicDummyEmbeddings(Embeddings, BaseModel):
    size: int

    def _get_embedding(self, text: str) -> list[float]:
        import numpy as np

        seed = abs(hash(text)) % (10**8)
        np.random.seed(seed)
        return list(np.random.normal(size=self.size))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)
