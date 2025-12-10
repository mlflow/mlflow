import asyncio
import json
import time
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
from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.utils import is_none_or_nan
from mlflow.genai.scorers.base import scorer
from mlflow.genai.utils.trace_utils import (
    _does_store_support_trace_linking,
    convert_predict_fn,
    create_minimal_trace,
    extract_available_tools_from_trace,
    extract_expectations_from_trace,
    extract_inputs_from_trace,
    extract_outputs_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
    extract_retrieval_context_from_trace,
    extract_tools_called_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    parse_tool_calls_from_trace,
    resolve_conversation_from_session,
)
from mlflow.tracing import set_span_chat_tools
from mlflow.tracing.utils import build_otel_context
from mlflow.types.chat import ToolCallOutput

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


def test_convert_predict_fn_skip_validation(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "true")

    call_count = 0

    def dummy_predict_fn(question: str, context: str):
        nonlocal call_count
        call_count += 1
        return question + context

    sample_input = {"question": "test", "context": "test"}
    converted_fn = convert_predict_fn(dummy_predict_fn, sample_input)
    # Predict function should not be validated when the env var is set to True
    assert call_count == 0

    # converted function takes a single 'request' argument
    result = converted_fn(request=sample_input)
    assert result == "testtest"


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


def test_does_store_support_trace_linking():
    test_trace = Trace(info=create_test_trace_info(trace_id="tr-123"), data=TraceData(spans=[]))

    # Databricks backend support trace linking
    assert _does_store_support_trace_linking(
        tracking_uri="databricks",
        trace=test_trace,
        run_id="run-123",
    )

    assert _does_store_support_trace_linking(
        tracking_uri="databricks://test",
        trace=test_trace,
        run_id="run-123",
    )

    mock_client = mock.MagicMock()
    with mock.patch("mlflow.genai.utils.trace_utils.MlflowClient", return_value=mock_client):
        # SQLAlchemy backend support trace linking
        mock_client.link_traces_to_run.side_effect = None

        assert _does_store_support_trace_linking(
            tracking_uri="sqlalchemy://test",
            trace=test_trace,
            run_id="run-123",
        )

        # File store doesn't support trace linking
        mock_client.link_traces_to_run.side_effect = Exception("Test error")

        assert not _does_store_support_trace_linking(
            tracking_uri="file://test",
            trace=test_trace,
            run_id="run-123",
        )

        # Result should be cached per tracking URI
        mock_client.reset_mock()
        mock_client.link_traces_to_run.side_effect = None
        for _ in range(10):
            assert _does_store_support_trace_linking(
                tracking_uri="sqlalchemy://test2",
                trace=test_trace,
                run_id="run-123",
            )
        mock_client.link_traces_to_run.assert_called_once()


def test_create_minimal_trace_restores_session_metadata():
    source = DatasetRecordSource(
        source_type=DatasetRecordSourceType.TRACE,
        source_data={"trace_id": "tr-original", "session_id": "session_1"},
    )

    eval_item = EvalItem(
        request_id="req-123",
        inputs={"question": "test"},
        outputs="answer",
        expectations={},
        source=source,
    )

    trace = create_minimal_trace(eval_item)

    # Verify session metadata was restored
    assert trace.info.trace_metadata.get("mlflow.trace.session") == "session_1"
    assert trace.data._get_root_span().inputs == {"question": "test"}
    assert trace.data._get_root_span().outputs == "answer"


def test_create_minimal_trace_without_source():
    eval_item = EvalItem(
        request_id="req-123",
        inputs={"question": "test"},
        outputs="answer",
        expectations={},
        source=None,
    )

    trace = create_minimal_trace(eval_item)

    # Should create trace successfully without session metadata
    assert trace is not None
    assert "mlflow.trace.session" not in trace.info.trace_metadata
    assert trace.data._get_root_span().inputs == {"question": "test"}
    assert trace.data._get_root_span().outputs == "answer"


def test_create_minimal_trace_with_source_but_no_session():
    source = DatasetRecordSource(
        source_type=DatasetRecordSourceType.TRACE,
        source_data={"trace_id": "tr-original"},  # No session_id
    )

    eval_item = EvalItem(
        request_id="req-123",
        inputs={"question": "test"},
        outputs="answer",
        expectations={},
        source=source,
    )

    trace = create_minimal_trace(eval_item)

    # Should work without session metadata
    assert trace is not None
    assert "mlflow.trace.session" not in trace.info.trace_metadata
    assert trace.data._get_root_span().inputs == {"question": "test"}
    assert trace.data._get_root_span().outputs == "answer"


def test_parse_tool_calls_from_trace():
    with mlflow.start_span(name="root") as root_span:
        root_span.set_inputs({"question": "What is the stock price?"})

        with mlflow.start_span(name="get_stock_price", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"symbol": "AAPL"})
            tool_span.set_outputs({"price": 150.0})

        with mlflow.start_span(name="get_market_cap", span_type=SpanType.TOOL) as tool_span2:
            tool_span2.set_inputs({"symbol": "AAPL"})
            tool_span2.set_outputs({"market_cap": "2.5T"})

        root_span.set_outputs("AAPL price is $150.")

    trace = mlflow.get_trace(root_span.trace_id)
    tool_messages = parse_tool_calls_from_trace(trace)

    assert len(tool_messages) == 2
    assert tool_messages[0] == {
        "role": "tool",
        "content": "Tool: get_stock_price\nInputs: {'symbol': 'AAPL'}\nOutputs: {'price': 150.0}",
    }
    assert tool_messages[1] == {
        "role": "tool",
        "content": (
            "Tool: get_market_cap\nInputs: {'symbol': 'AAPL'}\nOutputs: {'market_cap': '2.5T'}"
        ),
    }


def test_parse_tool_calls_from_trace_no_tools():
    with mlflow.start_span(name="root") as span:
        span.set_inputs({"question": "Hello"})
        span.set_outputs("Hi there")

    trace = mlflow.get_trace(span.trace_id)
    tool_messages = parse_tool_calls_from_trace(trace)

    assert tool_messages == []


def test_parse_tool_calls_from_trace_tool_without_outputs():
    with mlflow.start_span(name="root") as root_span:
        root_span.set_inputs({"query": "test"})

        with mlflow.start_span(name="my_tool", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"param": "value"})

        root_span.set_outputs("result")

    trace = mlflow.get_trace(root_span.trace_id)
    tool_messages = parse_tool_calls_from_trace(trace)

    assert len(tool_messages) == 1
    assert tool_messages[0] == {
        "role": "tool",
        "content": "Tool: my_tool\nInputs: {'param': 'value'}",
    }


def test_resolve_conversation_from_session():
    session_id = "test_session_resolve"
    traces = []

    with mlflow.start_span(name="turn_0") as span:
        span.set_inputs({"messages": [{"role": "user", "content": "What is AAPL price?"}]})
        span.set_outputs("AAPL is $150.")
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    traces.append(mlflow.get_trace(span.trace_id))

    with mlflow.start_span(name="turn_1") as span:
        span.set_inputs({"messages": [{"role": "user", "content": "How about MSFT?"}]})
        span.set_outputs("MSFT is $300.")
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    traces.append(mlflow.get_trace(span.trace_id))

    conversation = resolve_conversation_from_session(traces)

    assert len(conversation) == 4
    assert conversation[0] == {"role": "user", "content": "What is AAPL price?"}
    assert conversation[1] == {"role": "assistant", "content": "AAPL is $150."}
    assert conversation[2] == {"role": "user", "content": "How about MSFT?"}
    assert conversation[3] == {"role": "assistant", "content": "MSFT is $300."}


def test_resolve_conversation_from_session_with_tool_calls():
    session_id = "test_session_with_tools"
    traces = []

    with mlflow.start_span(name="turn_0") as root_span:
        root_span.set_inputs({"messages": [{"role": "user", "content": "Get AAPL price"}]})

        with mlflow.start_span(name="get_stock_price", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"symbol": "AAPL"})
            tool_span.set_outputs({"price": 150})

        root_span.set_outputs("AAPL is $150.")
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    traces.append(mlflow.get_trace(root_span.trace_id))

    conversation = resolve_conversation_from_session(traces, include_tool_calls=False)
    assert len(conversation) == 2
    assert conversation[0]["role"] == "user"
    assert conversation[1]["role"] == "assistant"

    conversation_with_tools = resolve_conversation_from_session(traces, include_tool_calls=True)
    assert len(conversation_with_tools) == 3
    assert conversation_with_tools[0] == {"role": "user", "content": "Get AAPL price"}
    assert conversation_with_tools[1] == {
        "role": "tool",
        "content": "Tool: get_stock_price\nInputs: {'symbol': 'AAPL'}\nOutputs: {'price': 150}",
    }
    assert conversation_with_tools[2] == {"role": "assistant", "content": "AAPL is $150."}


def test_resolve_conversation_from_session_empty():
    assert resolve_conversation_from_session([]) == []


def test_convert_predict_fn_async_function():
    async def async_predict_fn(request):
        await asyncio.sleep(0.01)
        return "async test response"

    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    converted_fn = convert_predict_fn(async_predict_fn, sample_input)

    result = converted_fn(request=sample_input)
    assert result == "async test response"

    traces = get_traces()
    assert len(traces) == 1
    purge_traces()


def test_evaluate_with_async_predict_fn():
    async def async_predict_fn(request):
        await asyncio.sleep(0.01)
        return "async test response"

    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    @scorer
    def dummy_scorer(inputs, outputs):
        return 0

    mlflow.genai.evaluate(
        data=[{"inputs": sample_input}],
        predict_fn=async_predict_fn,
        scorers=[dummy_scorer],
    )
    assert len(get_traces()) == 1
    purge_traces()


def test_convert_predict_fn_async_function_with_timeout(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT", "1")
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "true")

    async def slow_async_predict_fn(request):
        await asyncio.sleep(2)
        return "should timeout"

    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    converted_fn = convert_predict_fn(slow_async_predict_fn, sample_input)

    with pytest.raises(asyncio.TimeoutError):  # noqa: PT011
        converted_fn(request=sample_input)

    assert len(get_traces()) == 0


@pytest.mark.parametrize(
    ("span_type", "use_attribute", "tool_name", "tool_description"),
    [
        ("LLM", True, "get_weather", "Get current weather"),
        ("CHAT_MODEL", False, "search", "Search the web"),
    ],
)
def test_extract_available_tools_from_trace_basic(
    span_type, use_attribute, tool_name, tool_description
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": {"type": "object", "properties": {"param": {"type": "string"}}},
            },
        }
    ]

    with mlflow.start_span(name="test_span", span_type=span_type) as span:
        if use_attribute:
            set_span_chat_tools(span, tools)
            span.set_inputs({"prompt": "test"})
        else:
            span.set_inputs({"messages": [{"role": "user", "content": "test"}], "tools": tools})
        span.set_outputs({"response": "result"})

    trace = mlflow.get_trace(span.trace_id)
    extracted_tools = extract_available_tools_from_trace(trace)

    assert len(extracted_tools) == 1
    assert extracted_tools[0].function.name == tool_name
    assert extracted_tools[0].function.description == tool_description


def test_extract_available_tools_from_trace_with_multiple_spans():
    tool1 = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            },
        }
    ]

    tool2 = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                },
            },
        }
    ]

    with mlflow.start_span(name="parent") as parent:
        with mlflow.start_span(name="llm1", span_type="LLM") as span1:
            set_span_chat_tools(span1, tool1)

        with mlflow.start_span(name="llm2", span_type="CHAT_MODEL") as span2:
            set_span_chat_tools(span2, tool2)

    trace = mlflow.get_trace(parent.trace_id)
    extracted_tools = extract_available_tools_from_trace(trace)

    assert len(extracted_tools) == 2
    tool_names = [t.function.name for t in extracted_tools]
    assert "add" in tool_names
    assert "multiply" in tool_names


def test_extract_available_tools_from_trace_deduplication():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with mlflow.start_span(name="parent") as parent:
        with mlflow.start_span(name="llm1", span_type="LLM") as span1:
            set_span_chat_tools(span1, tools)

        with mlflow.start_span(name="llm2", span_type="LLM") as span2:
            set_span_chat_tools(span2, tools)  # Same tool

    trace = mlflow.get_trace(parent.trace_id)
    extracted_tools = extract_available_tools_from_trace(trace)

    # Should only return 1 tool despite being in 2 spans
    assert len(extracted_tools) == 1
    assert extracted_tools[0].function.name == "get_weather"


def test_extract_available_tools_from_trace_different_descriptions():
    tool1 = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    tool2 = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the database",  # Different description
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with mlflow.start_span(name="parent") as parent:
        with mlflow.start_span(name="llm1", span_type="LLM") as span1:
            set_span_chat_tools(span1, tool1)

        with mlflow.start_span(name="llm2", span_type="LLM") as span2:
            set_span_chat_tools(span2, tool2)

    trace = mlflow.get_trace(parent.trace_id)
    extracted_tools = extract_available_tools_from_trace(trace)

    # Should return 2 tools since descriptions are different
    assert len(extracted_tools) == 2
    descriptions = [t.function.description for t in extracted_tools]
    assert "Search the web" in descriptions
    assert "Search the database" in descriptions


@pytest.mark.parametrize(
    "trace_fixture",
    [
        None,
        Trace(info=create_test_trace_info(trace_id="tr-123"), data=None),
        Trace(info=create_test_trace_info(trace_id="tr-456"), data=TraceData(spans=[])),
    ],
)
def test_extract_available_tools_from_trace_returns_empty(trace_fixture):
    result = extract_available_tools_from_trace(trace_fixture)
    assert result == []


@pytest.mark.parametrize(
    ("has_valid_tool", "expected_count"),
    [
        (False, 0),  # Only invalid tools
        (True, 1),  # Mix of valid and invalid tools
    ],
)
def test_extract_available_tools_from_trace_with_invalid_tools(has_valid_tool, expected_count):
    with mlflow.start_span(name="parent") as parent:
        if has_valid_tool:
            valid_tool = [
                {
                    "type": "function",
                    "function": {
                        "name": "valid_tool",
                        "description": "A valid tool",
                    },
                }
            ]
            with mlflow.start_span(name="llm1", span_type="LLM") as span1:
                set_span_chat_tools(span1, valid_tool)

        with mlflow.start_span(name="llm2", span_type="LLM") as span2:
            span2.set_inputs(
                {
                    "messages": [{"role": "user", "content": "test"}],
                    "tools": [
                        {"invalid": "tool"},  # Missing required fields
                        {"type": "function"},  # Missing function field
                    ],
                }
            )

    trace = mlflow.get_trace(parent.trace_id)
    extracted_tools = extract_available_tools_from_trace(trace)

    assert len(extracted_tools) == expected_count
    if has_valid_tool:
        assert extracted_tools[0].function.name == "valid_tool"


def test_extract_tools_called_from_trace_basic():
    messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
        {"role": "assistant", "content": "It's sunny and 72°F in SF."},
    ]

    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        span.set_inputs({"messages": messages})
        span.set_attribute("mlflow.message.format", "openai")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ToolCallOutput)
    assert tool_calls[0].tool_call.id == "call_123"
    assert tool_calls[0].tool_call.function.name == "get_weather"
    assert tool_calls[0].tool_call.function.arguments == '{"location": "SF"}'
    assert tool_calls[0].output == "Sunny, 72°F"


def test_extract_tools_called_from_trace_multiple_tools():
    messages = [
        {"role": "user", "content": "Calculate 5+3 and 10*2"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"x": 10, "y": 2}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "8"},
        {"role": "tool", "tool_call_id": "call_2", "content": "20"},
    ]

    with mlflow.start_span(name="test_span", span_type="CHAT_MODEL") as span:
        span.set_inputs({"messages": messages})
        span.set_attribute("mlflow.message.format", "openai")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert len(tool_calls) == 2
    assert tool_calls[0].tool_call.id == "call_1"
    assert tool_calls[0].tool_call.function.name == "add"
    assert tool_calls[0].output == "8"
    assert tool_calls[1].tool_call.id == "call_2"
    assert tool_calls[1].tool_call.function.name == "multiply"
    assert tool_calls[1].output == "20"


def test_extract_tools_called_from_trace_without_output():
    messages = [
        {"role": "user", "content": "Get weather"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        },
    ]

    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        span.set_inputs({"messages": messages})
        span.set_attribute("mlflow.message.format", "openai")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call.id == "call_456"
    assert tool_calls[0].output == ""  # Empty string when output not found


def test_extract_tools_called_from_trace_uses_span_with_most_complete_tool_calls():
    # First span: incomplete conversation with only 1 tool call (no tool result)
    messages1 = [
        {"role": "user", "content": "First call"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "old_tool", "arguments": "{}"},
                }
            ],
        },
    ]

    # Second span: complete conversation with 2 tool calls and their results
    messages2 = [
        {"role": "user", "content": "First call"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "old_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_new",
                    "type": "function",
                    "function": {"name": "new_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_new", "content": "new result"},
    ]

    with mlflow.start_span(name="parent") as parent:
        with mlflow.start_span(name="llm1", span_type="LLM") as span1:
            span1.set_inputs({"messages": messages1})
            span1.set_attribute("mlflow.message.format", "openai")

        # Small delay to ensure different start times
        time.sleep(0.001)

        with mlflow.start_span(name="llm2", span_type="CHAT_MODEL") as span2:
            span2.set_inputs({"messages": messages2})
            span2.set_attribute("mlflow.message.format", "openai")

    trace = mlflow.get_trace(parent.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    # Should extract tool calls from span2 which has the most (2 tool calls)
    assert len(tool_calls) == 2
    assert tool_calls[0].tool_call.id == "call_old"
    assert tool_calls[0].tool_call.function.name == "old_tool"
    assert tool_calls[0].output == "old result"
    assert tool_calls[1].tool_call.id == "call_new"
    assert tool_calls[1].tool_call.function.name == "new_tool"
    assert tool_calls[1].output == "new result"


def test_extract_tools_called_from_trace_no_llm_spans():
    with mlflow.start_span(name="test_span", span_type="CHAIN") as span:
        span.set_inputs({"query": "test"})

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert tool_calls == []


def test_extract_tools_called_from_trace_no_inputs():
    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        pass  # Don't set inputs

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert tool_calls == []


def test_extract_tools_called_from_trace_no_tool_calls():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        span.set_inputs({"messages": messages})
        span.set_attribute("mlflow.message.format", "openai")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert tool_calls == []


def test_extract_tools_called_from_trace_langchain_format():
    messages = [
        {"type": "human", "content": "Search for Python"},
        {
            "type": "ai",
            "content": "",
            "tool_calls": [
                {"id": "tc_1", "name": "web_search", "args": {"query": "Python"}},
            ],
        },
        {"type": "tool", "content": "Found Python docs", "tool_call_id": "tc_1"},
    ]

    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        span.set_inputs({"messages": messages})
        span.set_attribute("mlflow.message.format", "langchain")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call.function.name == "web_search"
    assert tool_calls[0].output == "Found Python docs"


def test_extract_tools_called_from_trace_openai_agent_format():
    messages = [
        {"role": "user", "content": "Calculate 10+20"},
        {
            "type": "function_call",
            "call_id": "fc_1",
            "name": "calculate",
            "arguments": '{"expr": "10+20"}',
        },
        {"type": "function_call_output", "call_id": "fc_1", "output": "30"},
    ]

    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        span.set_inputs(messages)
        span.set_attribute("mlflow.message.format", "openai-agent")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call.id == "fc_1"
    assert tool_calls[0].tool_call.function.name == "calculate"
    assert tool_calls[0].output == "30"


def test_extract_tools_called_from_trace_invalid_tool_call():
    messages = [
        {"role": "user", "content": "Test"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "valid_call",
                    "type": "function",
                    "function": {"name": "valid_tool", "arguments": "{}"},
                },
                {"invalid": "tool_call"},  # Missing required fields
            ],
        },
        {"role": "tool", "tool_call_id": "valid_call", "content": "result"},
    ]

    with mlflow.start_span(name="test_span", span_type="LLM") as span:
        span.set_inputs({"messages": messages})
        span.set_attribute("mlflow.message.format", "openai")

    trace = mlflow.get_trace(span.trace_id)
    tool_calls = extract_tools_called_from_trace(trace)

    # Only valid tool call should be extracted
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call.id == "valid_call"
