import json
from typing import Any
from unittest import mock

import httpx
import openai
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.genai.utils.trace_utils import convert_predict_fn, extract_retrieval_context_from_trace
from mlflow.tracing.utils import build_otel_context

from tests.tracing.helper import create_test_trace_info, get_traces, purge_traces


def httpx_send_patch(request, *args, **kwargs):
    return httpx.Response(
        status_code=200,
        request=request,
        json={
            "id": "chatcmpl-Ax4UAd5xf32KjgLkS1SEEY9oorI9m",
            "object": "chat.completion",
            "created": 1738641958,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "test",
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        },
    )


def get_openai_predict_fn(with_tracing=False):
    if with_tracing:
        mlflow.openai.autolog()

    def predict_fn(request):
        with mock.patch("httpx.Client.send", side_effect=httpx_send_patch):
            response = openai.OpenAI().chat.completions.create(
                messages=request["messages"],
                model="gpt-4o-mini",
            )
            return response.choices[0].message.content

    return predict_fn


def get_dummy_predict_fn(with_tracing=False):
    def predict_fn(request):
        return "test"

    if with_tracing:
        return mlflow.trace(predict_fn)

    return predict_fn


@pytest.fixture
def mock_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_api_key")


@pytest.mark.usefixtures("mock_openai_env")
@pytest.mark.parametrize(
    ("predict_fn_generator", "with_tracing"),
    [
        (get_dummy_predict_fn, False),
        (get_dummy_predict_fn, True),
        (get_openai_predict_fn, False),
        (get_openai_predict_fn, True),
    ],
    ids=[
        "dummy predict_fn without tracing",
        "dummy predict_fn with tracing",
        "openai predict_fn without tracing",
        "openai predict_fn with tracing",
    ],
)
def test_convert_predict_fn(predict_fn_generator, with_tracing):
    predict_fn = predict_fn_generator(with_tracing=with_tracing)
    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    # predict_fn is callable as is
    result = predict_fn(**sample_input)
    assert result == "test"
    assert len(get_traces()) == (1 if with_tracing else 0)
    purge_traces()

    converted_fn = convert_predict_fn(predict_fn, sample_input)

    # converted function takes a single 'request' argument
    result = converted_fn(request=sample_input)
    assert result == "test"

    # Trace should be generated
    assert len(get_traces()) == 1


def create_span(
    span_id: int,
    parent_id: int,
    span_type: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> Span:
    otel_span = OTelReadableSpan(
        name="test",
        context=build_otel_context(123, span_id),
        parent=build_otel_context(123, parent_id) if parent_id else None,
        start_time=100,
        end_time=200,
        attributes={
            "mlflow.spanInputs": json.dumps(inputs),
            "mlflow.spanOutputs": json.dumps(outputs),
            "mlflow.spanType": json.dumps(span_type),
        },
    )
    return Span(otel_span)


@pytest.mark.parametrize(
    ("spans", "expected_retrieval_context"),
    [
        # multiple retrieval steps - only take the last top-level one
        (
            [
                create_span(
                    span_id=1,
                    parent_id=None,  # root span
                    inputs="question",
                    outputs={"generations": [[{"text": "some text"}]]},
                    span_type=SpanType.LLM,
                ),
                create_span(
                    span_id=2,
                    parent_id=1,
                    inputs="This should be ignored",
                    outputs=[
                        {
                            "page_content": "document content 3",
                            "metadata": {
                                "doc_uri": "uri3",
                                "chunk_id": "3",
                            },
                            "type": "Document",
                        },
                    ],
                    span_type=SpanType.RETRIEVER,
                ),
                create_span(
                    span_id=3,
                    parent_id=1,
                    inputs="What is the capital of France?",
                    outputs=[
                        {
                            "page_content": "document content 1",
                            "metadata": {
                                "doc_uri": "uri1",
                                "chunk_id": "1",
                            },
                            "type": "Document",
                        },
                        {
                            "page_content": "document content 2",
                            "metadata": {
                                "doc_uri": "uri2",
                                "chunk_id": "2",
                            },
                            "type": "Document",
                        },
                    ],
                    span_type=SpanType.RETRIEVER,
                ),
                create_span(
                    span_id=4,
                    parent_id=3,
                    inputs="This should be ignored because it's not a top-level retrieval span",
                    outputs=[
                        {
                            "page_content": "document content 4",
                            "metadata": {
                                "doc_uri": "uri4",
                                "chunk_id": "4",
                            },
                            "type": "Document",
                        },
                    ],
                    span_type=SpanType.RETRIEVER,
                ),
            ],
            [
                {
                    "doc_uri": "uri1",
                    "content": "document content 1",
                },
                {
                    "doc_uri": "uri2",
                    "content": "document content 2",
                },
            ],
        ),
        # one retrieval step
        (
            [
                create_span(
                    span_id=1,
                    parent_id=None,
                    inputs="What is the capital of France?",
                    outputs=[
                        {
                            "page_content": "document content 1",
                            "metadata": {
                                "doc_uri": "uri1",
                                "chunk_id": "1",
                            },
                            "type": "Document",
                        },
                        # missing doc_uri
                        {
                            "page_content": "document content 2",
                            "metadata": {
                                "chunk_id": "2",
                            },
                            "type": "Document",
                        },
                        # missing content
                        {
                            "metadata": {
                                "doc_uri": "uri3",
                                "chunk_id": "3",
                            },
                            "type": "Document",
                        },
                        # missing metadata
                        {
                            "page_content": "document content 4",
                            "type": "Document",
                        },
                    ],
                    span_type=SpanType.RETRIEVER,
                ),
            ],
            [
                {
                    "doc_uri": "uri1",
                    "content": "document content 1",
                },
                {
                    "content": "document content 2",
                },
                {
                    "content": None,
                    "doc_uri": "uri3",
                },
                {
                    "content": "document content 4",
                },
            ],
        ),
        # one retrieval step - empty retrieval span outputs
        (
            [
                create_span(
                    span_id=1,
                    parent_id=None,
                    inputs="What is the capital of France?",
                    outputs=[],
                    span_type=SpanType.RETRIEVER,
                ),
            ],
            [],
        ),
        # one retrieval step - wrong format retrieval span outputs
        (
            [
                create_span(
                    span_id=1,
                    parent_id=None,
                    inputs="What is the capital of France?",
                    outputs=["wrong output", "should be ignored"],
                    span_type=SpanType.RETRIEVER,
                ),
            ],
            [],
        ),
        # no retrieval steps
        (
            [
                create_span(
                    span_id=1,
                    parent_id=None,
                    inputs="What is the capital of France?",
                    outputs=[{"text": "some text"}],
                    span_type=SpanType.LLM,
                ),
            ],
            None,
        ),
        # None trace
        (
            None,
            None,
        ),
    ],
)
def test_get_retrieval_context_from_trace(spans, expected_retrieval_context):
    """Test traces.extract_retrieval_context_from_trace."""
    trace = Trace(info=create_test_trace_info(trace_id="tr-123"), data=TraceData(spans=spans))
    assert extract_retrieval_context_from_trace(trace) == expected_retrieval_context
