import json
from typing import Any
from unittest import mock

import httpx
import numpy as np
import openai
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities.assessment import Expectation
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.genai.evaluation.utils import is_none_or_nan
from mlflow.genai.scorers.base import scorer
from mlflow.genai.utils.trace_utils import (
    convert_predict_fn,
    extract_expectations_from_trace,
    extract_inputs_from_trace,
    extract_outputs_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
)
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
    ("predict_fn_generator", "with_tracing", "should_be_wrapped"),
    [
        (get_dummy_predict_fn, False, True),
        # If the function is already traced, it should not be wrapped with @mlflow.trace.
        (get_dummy_predict_fn, True, False),
        # OpenAI autologging is automatically enabled during evaluation,
        # so we don't need to wrap the function with @mlflow.trace.
        (get_openai_predict_fn, False, False),
        (get_openai_predict_fn, True, False),
    ],
    ids=[
        "dummy predict_fn without tracing",
        "dummy predict_fn with tracing",
        "openai predict_fn without tracing",
        "openai predict_fn with tracing",
    ],
)
def test_convert_predict_fn(predict_fn_generator, with_tracing, should_be_wrapped):
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

    # Trace should be generated if decorated or wrapped with @mlflow.trace
    assert len(get_traces()) == (1 if with_tracing or should_be_wrapped else 0)
    purge_traces()

    # All function should generate a trace when executed through mlflow.genai.evaluate
    @scorer
    def dummy_scorer(inputs, outputs):
        return 0

    mlflow.genai.evaluate(
        data=[{"inputs": sample_input}],
        predict_fn=predict_fn,
        scorers=[dummy_scorer],
    )
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
                    inputs="What is the capital of France?",
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
            {
                "0000000000000002": [
                    {
                        "doc_uri": "uri3",
                        "content": "document content 3",
                    },
                ],
                "0000000000000003": [
                    {
                        "doc_uri": "uri1",
                        "content": "document content 1",
                    },
                    {
                        "doc_uri": "uri2",
                        "content": "document content 2",
                    },
                ],
            },
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
            {
                "0000000000000001": [
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
            },
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
            {"0000000000000001": []},
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
            {"0000000000000001": []},
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
            {},
        ),
        # None trace
        (
            None,
            {},
        ),
    ],
)
def test_get_retrieval_context_from_trace(spans, expected_retrieval_context):
    """Test traces.extract_retrieval_context_from_trace."""
    trace = Trace(info=create_test_trace_info(trace_id="tr-123"), data=TraceData(spans=spans))
    assert extract_retrieval_context_from_trace(trace) == expected_retrieval_context


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        # String input
        ("Hello world", "Hello world"),
        # Chat completion/ChatModel/ChatAgent request
        (
            {"messages": [{"role": "user", "content": "User message"}]},
            "User message",
        ),
        # Multi-turn messages
        (
            {
                "messages": [
                    {"role": "assistant", "content": "First"},
                    {"role": "user", "content": "Second"},
                ]
            },
            '[{"role": "assistant", "content": "First"}, {"role": "user", "content": "Second"}]',
        ),
        # Empty dict input
        (
            {},
            "{}",
        ),
        # Dict input
        (
            {"unsupported_key": "value"},
            "{'unsupported_key': 'value'}",
        ),
        # Non-standard messages
        (
            {
                "messages": [
                    {"role": "assistant", "k": "First"},
                    {"role": "user", "k": "Second"},
                ]
            },
            "{'messages': [{'role': 'assistant', 'k': 'First'}, {'role': 'user', 'k': 'Second'}]}",
        ),
    ],
)
def test_parse_inputs_to_str(input_data, expected):
    assert parse_inputs_to_str(input_data) == expected


@pytest.mark.parametrize(
    ("output_data", "expected"),
    [
        # String output
        ("Output string", "Output string"),
        # Chat completion/ChatModel response
        (
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Output content",
                        },
                    }
                ]
            },
            "Output content",
        ),
        # ChatAgent response with multiple messages
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Input content",
                    },
                    {
                        "role": "assistant",
                        "content": "Intermediate Output content",
                    },
                    {
                        "role": "user",
                        "content": "Intermediate Input content",
                    },
                    {
                        "role": "assistant",
                        "content": "Output content",
                    },
                ]
            },
            "Output content",
        ),
        # List of strings
        (["Response content"], "Response content"),
        # ChatAgent response with multiple messages
        (
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Output content",
                            },
                        }
                    ]
                }
            ],
            "Output content",
        ),
        # List of direct string response
        (
            {"unsupported_key": "value"},
            '{"unsupported_key": "value"}',
        ),
        # Handle custom messages array format
        (
            {"messages": ["a", "b", "c"]},
            '{"messages": ["a", "b", "c"]}',
        ),
    ],
)
def test_parse_outputs_to_str(output_data, expected):
    assert parse_outputs_to_str(output_data) == expected


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (None, True),
        (np.nan, True),
        (float("nan"), True),
        ("Not NaN", False),
        (123, False),
        ([], False),
        ({}, False),
        (0.0, False),
        (1.5, False),
    ],
)
def test_is_none_or_nan(input_value, expected):
    assert is_none_or_nan(input_value) == expected


def test_extract_expectations_from_trace_with_source_filter():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is an open source platform"})

    trace_id = span.trace_id

    human_expectation = Expectation(
        name="human_expectation",
        value={"expected": "Answer from human"},
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=human_expectation)

    llm_expectation = Expectation(
        name="llm_expectation",
        value="LLM generated expectation",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=llm_expectation)

    code_expectation = Expectation(
        name="code_expectation",
        value=42,
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=code_expectation)

    trace = mlflow.get_trace(trace_id)

    result = extract_expectations_from_trace(trace, source=None)
    assert result == {
        "human_expectation": {"expected": "Answer from human"},
        "llm_expectation": "LLM generated expectation",
        "code_expectation": 42,
    }

    result = extract_expectations_from_trace(trace, source="HUMAN")
    assert result == {"human_expectation": {"expected": "Answer from human"}}

    result = extract_expectations_from_trace(trace, source="LLM_JUDGE")
    assert result == {"llm_expectation": "LLM generated expectation"}

    result = extract_expectations_from_trace(trace, source="CODE")
    assert result == {"code_expectation": 42}

    result = extract_expectations_from_trace(trace, source="human")
    assert result == {"human_expectation": {"expected": "Answer from human"}}

    with pytest.raises(mlflow.exceptions.MlflowException, match="Invalid assessment source type"):
        extract_expectations_from_trace(trace, source="INVALID_SOURCE")


def test_extract_expectations_from_trace_returns_none_when_no_expectations():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is an open source platform"})

    trace = mlflow.get_trace(span.trace_id)

    result = extract_expectations_from_trace(trace)
    assert result is None

    result = extract_expectations_from_trace(trace, source="HUMAN")
    assert result is None


def test_extract_inputs_and_outputs_from_trace():
    test_inputs = {"question": "What is MLflow?", "context": "MLflow is a tool"}
    test_outputs = {"answer": "MLflow is an open source platform", "confidence": 0.95}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(test_inputs)
        span.set_outputs(test_outputs)

    trace = mlflow.get_trace(span.trace_id)

    assert extract_inputs_from_trace(trace) == test_inputs
    assert extract_outputs_from_trace(trace) == test_outputs

    trace_without_data = Trace(
        info=create_test_trace_info(trace_id="tr-123"), data=TraceData(spans=[])
    )
    assert extract_inputs_from_trace(trace_without_data) is None
    assert extract_outputs_from_trace(trace_without_data) is None


def test_extract_request_and_response_from_trace():
    test_inputs = {"messages": [{"role": "user", "content": "What is MLflow?"}]}
    test_outputs = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "MLflow is great"}}]
    }

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(test_inputs)
        span.set_outputs(test_outputs)

    trace = mlflow.get_trace(span.trace_id)

    assert extract_request_from_trace(trace) == "What is MLflow?"
    assert extract_response_from_trace(trace) == "MLflow is great"

    trace_without_data = Trace(
        info=create_test_trace_info(trace_id="tr-123"), data=TraceData(spans=[])
    )
    assert extract_request_from_trace(trace_without_data) is None
    assert extract_response_from_trace(trace_without_data) is None


def test_extract_request_and_response_with_string_inputs():
    test_inputs = "Simple string input"
    test_outputs = "Simple string output"

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(test_inputs)
        span.set_outputs(test_outputs)

    trace = mlflow.get_trace(span.trace_id)

    assert extract_request_from_trace(trace) == "Simple string input"
    assert extract_response_from_trace(trace) == "Simple string output"
