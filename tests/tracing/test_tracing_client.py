import json
from unittest.mock import Mock, patch

import mlflow
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.client import TracingClient

from tests.tracing.helper import skip_when_testing_trace_sdk


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
