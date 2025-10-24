import json
from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_TRACING_SQL_WAREHOUSE_ID
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.client import TracingClient

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
def test_tracing_client_search_traces_with_model_id(monkeypatch, sql_warehouse_id):
    if sql_warehouse_id:
        monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, sql_warehouse_id)
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
