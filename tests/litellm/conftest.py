import importlib

import openai
import pytest

from tests.helper_functions import start_mock_openai_server


@pytest.fixture(autouse=True)
def set_envs(monkeypatch, mock_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai)
    importlib.reload(openai)


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url
