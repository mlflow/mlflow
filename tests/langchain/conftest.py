import importlib
from typing import List

import numpy as np
import openai
import pytest
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel

from tests.helper_functions import start_mock_openai_server


@pytest.fixture(autouse=True)
def set_envs(monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
            "SERPAPI_API_KEY": "test",
        }
    )
    importlib.reload(openai)


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url


# Define a special embedding for testing
class DeterministicDummyEmbeddings(Embeddings, BaseModel):
    size: int

    def _get_embedding(self, text: str) -> List[float]:
        seed = abs(hash(text)) % (10**8)
        np.random.seed(seed)
        return list(np.random.normal(size=self.size))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)
