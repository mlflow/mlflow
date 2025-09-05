import importlib

import openai
import pytest
import pytest_asyncio
from opentelemetry import trace as trace_api
from opentelemetry.util._once import Once

from tests.helper_functions import start_mock_openai_server
from tests.tracing.helper import (
    reset_autolog_state,  # noqa: F401
)


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


@pytest_asyncio.fixture(autouse=True)
async def post_test_tracing_cleanup():
    yield

    trace_api._TRACER_PROVIDER_SET_ONCE = Once()
    trace_api._TRACER_PROVIDER = None
