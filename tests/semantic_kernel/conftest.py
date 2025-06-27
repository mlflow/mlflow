import importlib

import openai
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.util._once import Once

import mlflow

from tests.helper_functions import start_mock_openai_server
from tests.tracing.helper import (
    reset_autolog_state,  # noqa: F401
)


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


@pytest.fixture(autouse=True)
def reset_autolog(reset_autolog_state):
    # Apply the reset_autolog_state fixture to all tests for LangChain
    return


@pytest.fixture
def dummy_otel_span_processor():
    """A dummy OpenTelemetry span processor that does nothing."""

    class DummySpanExporter:
        # Dummy NoOp exporter that does nothing, because OTel span processor requires an exporter
        def on_end(self, *args, **kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class DummySpanProcessor(SimpleSpanProcessor):
        def __init__(self, span_exporter):
            self.span_exporter = DummySpanExporter()

        def on_start(self, *args, **kwargs):
            pass

        def on_end(self, *args, **kwargs):
            pass

    return DummySpanProcessor(DummySpanExporter())


def _reset_tracing_globals() -> None:
    trace_api._TRACER_PROVIDER_SET_ONCE = Once()
    trace_api._TRACER_PROVIDER = None
    trace_api._PROXY_TRACER_PROVIDER = trace_api.ProxyTracerProvider()

    mlflow.tracing.provider.reset()


@pytest.fixture(autouse=True)
def reset_tracing_globals():
    _reset_tracing_globals()
    yield
    _reset_tracing_globals()
