"""Tests for mlflow.genai.scorers.inspect_ai.utils.

Verifies that MLflow eval inputs, traces, and sessions are correctly converted
into Inspect AI task payloads, including graceful handling of missing or None data.
"""
from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.inspect_ai.utils import (
    map_scorer_inputs_to_inspectai_payload,
    map_session_to_inspectai_conversational_payload,
)


def test_check_inspectai_installed_success():
    """Payload construction succeeds silently when inspectai is available."""
    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"):
        
        map_scorer_inputs_to_inspectai_payload(
            metric_name="test_metric", inputs="test input", outputs="test output"
        )


def test_check_inspectai_not_installed_raises_error():
    """MlflowException is raised with a helpful message when inspectai is missing."""
    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed") as mock_check:
        mock_check.side_effect = MlflowException(
            "Inspect AI scorers require the 'inspectai' package"
        )
        with pytest.raises(MlflowException, match="Inspect AI scorers require"):
            map_scorer_inputs_to_inspectai_payload(
                metric_name="test_metric", inputs="test input", outputs="test output"
            )


def test_map_scorer_inputs_basic_payload():
    """Basic inputs/outputs/expectations are stringified and placed in the correct payload keys."""
    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"):
        with patch("mlflow.genai.scorers.inspect_ai.utils.resolve_inputs_from_trace") as mock_inputs, \
             patch("mlflow.genai.scorers.inspect_ai.utils.resolve_outputs_from_trace") as mock_outputs, \
             patch("mlflow.genai.scorers.inspect_ai.utils.resolve_expectations_from_trace") as mock_exp, \
             patch("mlflow.genai.scorers.inspect_ai.utils.parse_inputs_to_str") as mock_parse_in, \
             patch("mlflow.genai.scorers.inspect_ai.utils.parse_outputs_to_str") as mock_parse_out:
            
            mock_inputs.return_value = "test input"
            mock_outputs.return_value = "test output"
            mock_exp.return_value = {"key": "value"}
            mock_parse_in.return_value = "parsed input"
            mock_parse_out.return_value = "parsed output"

            payload = map_scorer_inputs_to_inspectai_payload(
                metric_name="test_metric",
                inputs="test input",
                outputs="test output",
                expectations={"key": "value"},
            )

            assert payload["metric_name"] == "test_metric"
            assert payload["input"] == "parsed input"
            assert payload["output"] == "parsed output"
            assert payload["expectations"] == {"key": "value"}


def test_map_scorer_inputs_includes_trace_data():
    """When a trace is provided, retrieval context, metadata, and tags are added to the payload."""
    trace = Mock()
    trace.info.trace_metadata = {"trace_key": "trace_value"}
    trace.info.tags = {"tag_key": "tag_value"}

    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"), \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_inputs_from_trace") as mock_inputs, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_outputs_from_trace") as mock_outputs, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_expectations_from_trace") as mock_exp, \
         patch("mlflow.genai.scorers.inspect_ai.utils.parse_inputs_to_str") as mock_parse_in, \
         patch("mlflow.genai.scorers.inspect_ai.utils.parse_outputs_to_str") as mock_parse_out, \
         patch("mlflow.genai.scorers.inspect_ai.utils.extract_retrieval_context_from_trace") as mock_retrieval:

        mock_inputs.return_value = "input"
        mock_outputs.return_value = "output"
        mock_exp.return_value = None
        mock_parse_in.return_value = "input"
        mock_parse_out.return_value = "output"
        mock_retrieval.return_value = {"docs": ["doc1", "doc2"]}

        payload = map_scorer_inputs_to_inspectai_payload(
            metric_name="test_metric",
            inputs="input",
            outputs="output",
            trace=trace,
        )

        assert "retrieval_context" in payload
        assert "additional_metadata" in payload
        assert "tags" in payload
        assert payload["additional_metadata"] == {"trace_key": "trace_value"}
        assert payload["tags"] == {"tag_key": "tag_value"}


def test_map_scorer_inputs_handles_missing_trace_data():
    """Exceptions during trace field extraction are caught; payload receives empty dicts instead."""
    trace = Mock()
    trace.info.trace_metadata = None
    trace.info.tags = None

    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"), \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_inputs_from_trace") as mock_inputs, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_outputs_from_trace") as mock_outputs, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_expectations_from_trace") as mock_exp, \
         patch("mlflow.genai.scorers.inspect_ai.utils.parse_inputs_to_str") as mock_parse_in, \
         patch("mlflow.genai.scorers.inspect_ai.utils.parse_outputs_to_str") as mock_parse_out, \
         patch("mlflow.genai.scorers.inspect_ai.utils.extract_retrieval_context_from_trace") as mock_retrieval:

        mock_inputs.return_value = "input"
        mock_outputs.return_value = "output"
        mock_exp.return_value = None
        mock_parse_in.return_value = "input"
        mock_parse_out.return_value = "output"
        mock_retrieval.side_effect = Exception("Missing data")

        payload = map_scorer_inputs_to_inspectai_payload(
            metric_name="test_metric",
            inputs="input",
            outputs="output",
            trace=trace,
        )

        assert payload["retrieval_context"] == {}
        assert payload["additional_metadata"] == {}
        assert payload["tags"] == {}


def test_map_scorer_inputs_none_values():
    """None inputs/outputs/expectations produce None payload values without raising."""
    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"), \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_inputs_from_trace") as mock_inputs, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_outputs_from_trace") as mock_outputs, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_expectations_from_trace") as mock_exp:

        mock_inputs.return_value = None
        mock_outputs.return_value = None
        mock_exp.return_value = None

        payload = map_scorer_inputs_to_inspectai_payload(
            metric_name="test_metric",
            inputs=None,
            outputs=None,
        )

        assert payload["input"] is None
        assert payload["output"] is None
        assert payload["expectations"] is None


def test_map_session_to_conversational_payload():
    """Session traces are converted to a conversation list with metric_name and expectations."""
    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"), \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_expectations_from_session") as mock_exp, \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_conversation_from_session") as mock_conv:

        mock_exp.return_value = {"key": "value"}
        mock_conv.return_value = [
            {"role": "user", "content": "message 1"},
            {"role": "assistant", "content": "response 1"},
        ]

        session = [Mock(), Mock()]
        payload = map_session_to_inspectai_conversational_payload(
            metric_name="multi_turn_task",
            session=session,
            expectations={"key": "value"},
        )

        assert payload["metric_name"] == "multi_turn_task"
        assert payload["expectations"] == {"key": "value"}
        assert "conversation" in payload or "messages" in payload


def test_map_session_to_conversational_payload_with_options():
    """include_tool_calls and include_timing flags are forwarded to the conversation resolver."""
    with patch("mlflow.genai.scorers.inspect_ai.utils._check_inspectai_installed"), \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_expectations_from_session"), \
         patch("mlflow.genai.scorers.inspect_ai.utils.resolve_conversation_from_session") as mock_conv:

        mock_conv.return_value = []

        session = [Mock()]
        payload = map_session_to_inspectai_conversational_payload(
            metric_name="multi_turn_task",
            session=session,
            include_tool_calls=True,
            include_timing=True,
        )

        assert isinstance(payload, dict)
        assert "metric_name" in payload
