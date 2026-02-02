import json
import uuid
from unittest.mock import Mock, patch

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities.span import create_mlflow_span
from mlflow.environment_variables import MLFLOW_TRACING_SQL_WAREHOUSE_ID
from mlflow.exceptions import MlflowException
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpansLocation, TraceMetadataKey, TraceSizeStatsKey, TraceTagKey
from mlflow.tracing.utils import TraceJSONEncoder

from tests.tracing.helper import skip_when_testing_trace_sdk


def test_get_trace_v4():
    mock_store = Mock()
    mock_store.batch_get_traces.return_value = ["dummy_trace"]

    with patch("mlflow.tracing.client._get_store", return_value=mock_store):
        client = TracingClient()
        trace = client.get_trace("trace:/catalog.schema/1234567890")

    assert trace == "dummy_trace"
    mock_store.batch_get_traces.assert_called_once_with(
        ["trace:/catalog.schema/1234567890"], "catalog.schema"
    )


def test_get_trace_v4_retry():
    mock_store = Mock()
    mock_store.batch_get_traces.side_effect = [[], ["dummy_trace"]]

    with patch("mlflow.tracing.client._get_store", return_value=mock_store):
        client = TracingClient()
        trace = client.get_trace("trace:/catalog.schema/1234567890")

    assert trace == "dummy_trace"
    assert mock_store.batch_get_traces.call_count == 2


@skip_when_testing_trace_sdk
def test_tracing_client_link_prompt_versions_to_trace():
    with mlflow.start_run():
        # Register a prompt
        prompt_version = mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

        # Create a trace
        with mlflow.start_span("test_span"):
            trace_id = mlflow.get_active_trace_id()

        # Link prompts to trace
        client = TracingClient()
        client.link_prompt_versions_to_trace(trace_id, [prompt_version])

        # Verify the linked prompts tag was set
        trace = mlflow.get_trace(trace_id)
        assert "mlflow.linkedPrompts" in trace.info.tags

        # Parse and verify the linked prompts
        linked_prompts = json.loads(trace.info.tags["mlflow.linkedPrompts"])
        assert len(linked_prompts) == 1
        assert linked_prompts[0]["name"] == "test_prompt"
        assert linked_prompts[0]["version"] == "1"


def test_tracing_client_calculate_trace_filter_correlation():
    mock_store = Mock()

    expected_result = TraceFilterCorrelationResult(
        npmi=0.456,
        npmi_smoothed=0.445,
        filter1_count=100,
        filter2_count=80,
        joint_count=50,
        total_count=200,
    )
    mock_store.calculate_trace_filter_correlation.return_value = expected_result

    with patch("mlflow.tracing.client._get_store", return_value=mock_store):
        client = TracingClient()

        result = client.calculate_trace_filter_correlation(
            experiment_ids=["123", "456"],
            filter_string1="span.type = 'LLM'",
            filter_string2="feedback.quality > 0.8",
            base_filter="request_time > 1000",
        )

        mock_store.calculate_trace_filter_correlation.assert_called_once_with(
            experiment_ids=["123", "456"],
            filter_string1="span.type = 'LLM'",
            filter_string2="feedback.quality > 0.8",
            base_filter="request_time > 1000",
        )

        assert result == expected_result
        assert result.npmi == 0.456
        assert result.npmi_smoothed == 0.445
        assert result.filter1_count == 100
        assert result.filter2_count == 80
        assert result.joint_count == 50
        assert result.total_count == 200


def test_tracing_client_calculate_trace_filter_correlation_without_base_filter():
    mock_store = Mock()

    expected_result = TraceFilterCorrelationResult(
        npmi=float("nan"),
        npmi_smoothed=None,
        filter1_count=0,
        filter2_count=0,
        joint_count=0,
        total_count=100,
    )
    mock_store.calculate_trace_filter_correlation.return_value = expected_result

    with patch("mlflow.tracing.client._get_store", return_value=mock_store):
        client = TracingClient()

        result = client.calculate_trace_filter_correlation(
            experiment_ids=["789"],
            filter_string1="error = true",
            filter_string2="duration > 5000",
        )

        mock_store.calculate_trace_filter_correlation.assert_called_once_with(
            experiment_ids=["789"],
            filter_string1="error = true",
            filter_string2="duration > 5000",
            base_filter=None,
        )

        assert result == expected_result
        assert result.filter1_count == 0
        assert result.filter2_count == 0
        assert result.joint_count == 0
        assert result.total_count == 100


@pytest.mark.parametrize("sql_warehouse_id", [None, "some-warehouse-id"])
def test_tracing_client_search_traces_with_model_id(monkeypatch, sql_warehouse_id: str | None):
    if sql_warehouse_id:
        monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, sql_warehouse_id)
    else:
        monkeypatch.delenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, raising=False)
    mock_store = Mock()
    mock_store.search_traces.return_value = ([], None)

    with patch("mlflow.tracing.client._get_store", return_value=mock_store):
        client = TracingClient()
        client.search_traces(model_id="model_id")

    mock_store.search_traces.assert_called_once_with(
        experiment_ids=None,
        filter_string="request_metadata.`mlflow.modelId` = 'model_id'"
        if sql_warehouse_id is None
        else None,
        max_results=SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by=None,
        page_token=None,
        model_id="model_id" if sql_warehouse_id else None,
        locations=None,
    )


@skip_when_testing_trace_sdk
def test_tracing_client_get_trace_with_database_stored_spans():
    client = TracingClient()

    experiment_id = mlflow.create_experiment("test")
    trace_id = f"tr-{uuid.uuid4().hex}"

    store = client.store

    otel_span = OTelReadableSpan(
        name="test_span",
        context=trace_api.SpanContext(
            trace_id=12345,
            span_id=111,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
            "llm.model_name": "test-model",
            "custom.attribute": "test-value",
        },
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        resource=None,
    )

    span = create_mlflow_span(otel_span, trace_id, "LLM")

    store.log_spans(experiment_id, [span])

    trace = client.get_trace(trace_id)

    assert trace.info.trace_id == trace_id
    assert trace.info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACKING_STORE

    assert len(trace.data.spans) == 1
    loaded_span = trace.data.spans[0]

    assert loaded_span.name == "test_span"
    assert loaded_span.trace_id == trace_id
    assert loaded_span.start_time_ns == 1_000_000_000
    assert loaded_span.end_time_ns == 2_000_000_000
    assert loaded_span.attributes.get("llm.model_name") == "test-model"
    assert loaded_span.attributes.get("custom.attribute") == "test-value"


@skip_when_testing_trace_sdk
def test_tracing_client_get_trace_error_handling():
    client = TracingClient()

    experiment_id = mlflow.create_experiment("test")
    trace_id = f"tr-{uuid.uuid4().hex}"

    store = client.store

    otel_span = OTelReadableSpan(
        name="test_span",
        context=trace_api.SpanContext(
            trace_id=12345,
            span_id=111,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
            "llm.model_name": "test-model",
            "custom.attribute": "test-value",
        },
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        resource=None,
    )

    span = create_mlflow_span(otel_span, trace_id, "LLM")

    store.log_spans(experiment_id, [span])
    trace = client.get_trace(trace_id)
    trace_info = trace.info
    trace_info.trace_metadata[TraceMetadataKey.SIZE_STATS] = json.dumps(
        {TraceSizeStatsKey.NUM_SPANS: 2}
    )
    store.start_trace(trace_info)

    with pytest.raises(
        MlflowException, match=rf"Trace with ID {trace_id} is not fully exported yet"
    ):
        client.get_trace(trace_id)
