import asyncio
from typing import List
from unittest.mock import ANY

import openai
import pytest
from llama_index.core import (
    Settings,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_status import TraceStatus
from mlflow.llama_index.tracer import MlflowEventHandler, MlflowSpanHandler
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch, mock_openai):
    """Set the LLM and Embedding model to the mock OpenAI server."""
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )
    # Need to reset the settings to reflect the new env variables
    monkeypatch.setattr(Settings, "llm", OpenAI())
    monkeypatch.setattr(Settings, "embed_model", OpenAIEmbedding())


@pytest.fixture(autouse=True)
def set_handlers():
    span_handler = MlflowSpanHandler(mlflow.MlflowClient())
    event_handler = MlflowEventHandler(span_handler)

    dsp = get_dispatcher()
    dsp.add_span_handler(span_handler)
    dsp.add_event_handler(event_handler)

    yield

    # Clean up
    dsp.span_handlers = []
    dsp.event_handlers = []


def _get_all_traces() -> List[Trace]:
    """Utility function to get all traces in the test experiment."""
    return mlflow.MlflowClient().search_traces(experiment_ids=[DEFAULT_EXPERIMENT_ID])


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_llm_complete(is_async):
    # By default llama-index uses "gpt-3.5-turbo" model that only has chat interface,
    # and llama-index redirects completion call to chat endpoint. We use non-chat
    # model here to test trace for completion.
    model_name = "gpt-3.5-turbo-instruct"
    llm = OpenAI(model=model_name)

    response = asyncio.run(llm.acomplete("Hello")) if is_async else llm.complete("Hello")
    assert response.text == "Hello"

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.acomplete" if is_async else "OpenAI.complete"

    attr = spans[0].attributes
    assert attr[SpanAttributeKey.SPAN_TYPE] == SpanType.LLM
    assert attr[SpanAttributeKey.INPUTS] == {"args": ["Hello"]}
    assert attr[SpanAttributeKey.OUTPUTS]["text"] == "Hello"
    assert attr["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert attr["prompt"] == "Hello"
    assert attr["invocation_params"]["model_name"] == model_name
    assert attr["model_dict"]["model"] == model_name


@pytest.mark.parametrize("is_async", [True, False])
def test_trace_llm_chat(is_async):
    llm = OpenAI()
    message = ChatMessage(role="system", content="Hello")

    response = asyncio.run(llm.achat([message])) if is_async else llm.chat([message])
    assert isinstance(response.message, ChatMessage)
    assert response.message.content == '[{"role": "system", "content": "Hello"}]'

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.achat" if is_async else "OpenAI.chat"

    attr = spans[0].attributes
    assert attr[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAT_MODEL
    assert attr[SpanAttributeKey.INPUTS] == {
        "messages": [{"role": "system", "content": "Hello", "additional_kwargs": {}}]
    }
    assert attr[SpanAttributeKey.OUTPUTS] == {
        "message": {
            "role": "assistant",
            "content": '[{"role": "system", "content": "Hello"}]',
            "additional_kwargs": {},
        },
        "raw": ANY,
        "delta": None,
        "logprobs": None,
        "additional_kwargs": {},
    }
    assert attr["usage"] == {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
    assert attr["invocation_params"]["model_name"] == llm.metadata.model_name
    assert attr["model_dict"]["model"] == llm.metadata.model_name


def test_trace_llm_error(monkeypatch):
    # Disable callback as it doesn't handle error response well
    monkeypatch.setattr(Settings, "callback_manager", None)

    llm = OpenAI(
        api_base="http://localhost:1234/invalid/base",
        timeout=1,
        max_retries=0,
    )

    with pytest.raises(openai.APIConnectionError, match="Connection error."):
        llm.chat([ChatMessage(role="system", content="Hello")])

    traces = _get_all_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.ERROR
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.chat"
    assert spans[0].attributes[SpanAttributeKey.SPAN_TYPE] == SpanType.CHAT_MODEL
    assert spans[0].attributes[SpanAttributeKey.INPUTS] == {
        "messages": [{"role": "system", "content": "Hello", "additional_kwargs": {}}]
    }
    assert SpanAttributeKey.OUTPUTS not in spans[0].attributes
    events = traces[0].data.spans[0].events
    assert len(events) == 1
    assert events[0].attributes["exception.message"] == "Connection error."


def test_trace_retriever():
    pass


def test_trace_agent():
    pass


def test_trace_query_engine():
    pass


def test_trace_query_engine_async():
    pass


def test_trace_chat_engine():
    pass


def test_trace_reranker():
    pass
