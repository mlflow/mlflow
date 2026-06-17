import json
import logging
import shutil
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli.traces import commands
from mlflow.entities import (
    AssessmentSourceType,
    MlflowExperimentLocation,
    Trace,
    TraceData,
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
)
from mlflow.store.entities.paged_list import PagedList


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress logging for all tests."""
    # Suppress logging
    original_root = logging.root.level
    original_mlflow = logging.getLogger("mlflow").level
    original_alembic = logging.getLogger("alembic").level

    logging.root.setLevel(logging.CRITICAL)
    logging.getLogger("mlflow").setLevel(logging.CRITICAL)
    logging.getLogger("alembic").setLevel(logging.CRITICAL)

    yield

    # Restore original logging levels
    logging.root.setLevel(original_root)
    logging.getLogger("mlflow").setLevel(original_mlflow)
    logging.getLogger("alembic").setLevel(original_alembic)


@pytest.fixture
def runner():
    """Provide a CLI runner for testing."""
    return CliRunner(catch_exceptions=False)


def test_commands_group_exists():
    assert commands.name == "traces"
    assert commands.help is not None


def test_search_command_params():
    search_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "search"), None)
    assert search_cmd is not None
    param_names = [p.name for p in search_cmd.params]
    assert "experiment_id" in param_names
    assert "filter_string" in param_names
    assert "max_results" in param_names
    assert "order_by" in param_names
    assert "page_token" in param_names
    assert "output" in param_names
    assert "extract_fields" in param_names


def test_get_command_params():
    get_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "get"), None)
    assert get_cmd is not None
    param_names = [p.name for p in get_cmd.params]
    assert "trace_id" in param_names
    assert "extract_fields" in param_names
    assert "jq_filter" in param_names


def test_assessment_source_type_choices():
    log_feedback_cmd = next(
        (cmd for cmd in commands.commands.values() if cmd.name == "log-feedback"), None
    )
    assert log_feedback_cmd is not None

    source_type_param = next(
        (param for param in log_feedback_cmd.params if param.name == "source_type"), None
    )
    assert source_type_param is not None
    assert AssessmentSourceType.HUMAN in source_type_param.type.choices
    assert AssessmentSourceType.LLM_JUDGE in source_type_param.type.choices
    assert AssessmentSourceType.CODE in source_type_param.type.choices


def test_search_command_with_fields(runner):
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=MlflowExperimentLocation(experiment_id="1"),
    )
    trace = Trace(
        info=TraceInfo(
            trace_id="tr-123",
            state=TraceState.OK,
            request_time=1700000000000,
            execution_duration=1234,
            request_preview="test request",
            response_preview="test response",
            trace_location=trace_location,
        ),
        data=TraceData(spans=[]),
    )

    mock_result = PagedList([trace], None)

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = mock_result
        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--extract-fields", "info.trace_id,info.state"],
        )

        assert result.exit_code == 0
        assert "tr-123" in result.output
        assert "OK" in result.output


def test_get_command_with_fields(runner):
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=MlflowExperimentLocation(experiment_id="1"),
    )
    trace = Trace(
        info=TraceInfo(
            trace_id="tr-123",
            state=TraceState.OK,
            trace_location=trace_location,
            request_time=1700000000000,
            execution_duration=1234,
        ),
        data=TraceData(spans=[]),
    )

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.get_trace.return_value = trace
        result = runner.invoke(
            commands,
            ["get", "--trace-id", "tr-123", "--extract-fields", "info.trace_id"],
        )

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json == {"info": {"trace_id": "tr-123"}}


# Span status mirrors the real OTLP-style to_dict() schema (status.code, not status_code).
_JQ_TRACE_DICT = {
    "info": {"trace_id": "tr-123", "state": "OK"},
    "data": {
        "spans": [
            {"name": "root", "status": {"code": "STATUS_CODE_OK"}},
            {"name": "retriever", "status": {"code": "STATUS_CODE_ERROR"}},
            {"name": "llm", "status": {"code": "STATUS_CODE_ERROR"}},
        ]
    },
}

requires_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq binary not installed")


@pytest.fixture
def jq_trace():
    trace = mock.Mock()
    trace.to_dict.return_value = _JQ_TRACE_DICT
    return trace


# The --jq path now delegates to mlflow.tracing.utils.apply_jq_to_trace, which fetches the
# trace via mlflow.tracing.client.TracingClient and resolves jq via mlflow.tracing.utils.shutil,
# so these tests patch there rather than at the CLI module.
@requires_jq
def test_get_command_with_jq_filter(runner, jq_trace):
    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client:
        mock_client.return_value.get_trace.return_value = jq_trace
        result = runner.invoke(
            commands,
            [
                "get",
                "--trace-id",
                "tr-123",
                "--jq",
                '[.data.spans[] | select(.status.code=="STATUS_CODE_ERROR") | .name]',
            ],
        )

    assert result.exit_code == 0
    assert json.loads(result.output) == ["retriever", "llm"]
    mock_client.return_value.get_trace.assert_called_once_with("tr-123")


@requires_jq
def test_get_command_with_jq_reshape(runner, jq_trace):
    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client:
        mock_client.return_value.get_trace.return_value = jq_trace
        result = runner.invoke(
            commands,
            [
                "get",
                "--trace-id",
                "tr-123",
                "--jq",
                "{id: .info.trace_id, count: (.data.spans | length)}",
            ],
        )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "tr-123", "count": 3}


@requires_jq
def test_get_command_jq_takes_precedence_over_extract_fields(runner, jq_trace):
    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client:
        mock_client.return_value.get_trace.return_value = jq_trace
        result = runner.invoke(
            commands,
            [
                "get",
                "--trace-id",
                "tr-123",
                "--extract-fields",
                "info.trace_id",
                "--jq",
                ".info.state",
            ],
        )

    assert result.exit_code == 0
    assert json.loads(result.output) == "OK"


@requires_jq
def test_get_command_jq_invalid_filter(runner, jq_trace):
    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client:
        mock_client.return_value.get_trace.return_value = jq_trace
        result = runner.invoke(
            commands,
            ["get", "--trace-id", "tr-123", "--jq", ".data.spans["],
            catch_exceptions=True,
        )

    assert result.exit_code != 0
    assert "jq error" in result.output


@requires_jq
def test_get_command_jq_empty_match(runner, jq_trace):
    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client:
        mock_client.return_value.get_trace.return_value = jq_trace
        result = runner.invoke(
            commands,
            ["get", "--trace-id", "tr-123", "--jq", '[.data.spans[] | select(.name=="missing")]'],
        )

    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_get_command_jq_missing_binary(runner, jq_trace):
    with (
        mock.patch("mlflow.tracing.client.TracingClient") as mock_client,
        mock.patch("mlflow.tracing.utils.shutil.which", return_value=None) as mock_which,
    ):
        mock_client.return_value.get_trace.return_value = jq_trace
        result = runner.invoke(
            commands,
            ["get", "--trace-id", "tr-123", "--jq", ".info.state"],
            catch_exceptions=True,
        )

    assert result.exit_code != 0
    assert "brew install jq" in result.output
    mock_which.assert_called_once_with("jq")


def test_delete_command(runner):
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.delete_traces.return_value = 5
        result = runner.invoke(
            commands,
            ["delete", "--experiment-id", "1", "--trace-ids", "tr-1,tr-2,tr-3"],
        )

        assert result.exit_code == 0
        assert "Deleted 5 trace(s)" in result.output


def test_field_validation_error(runner):
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=MlflowExperimentLocation(experiment_id="1"),
    )
    trace = Trace(
        info=TraceInfo(
            trace_id="tr-123",
            trace_location=trace_location,
            request_time=1700000000000,
            execution_duration=1234,
            state=TraceState.OK,
        ),
        data=TraceData(spans=[]),
    )

    mock_result = PagedList([trace], None)

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = mock_result
        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--extract-fields", "invalid.field"],
        )

        assert result.exit_code != 0
        assert "Invalid field path" in result.output
        assert "--verbose" in result.output


def test_field_validation_error_verbose_mode(runner):
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=MlflowExperimentLocation(experiment_id="1"),
    )
    trace = Trace(
        info=TraceInfo(
            trace_id="tr-123",
            state=TraceState.OK,
            request_time=1700000000000,
            trace_location=trace_location,
            execution_duration=1234,
        ),
        data=TraceData(spans=[]),
    )

    mock_result = PagedList([trace], None)

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = mock_result
        result = runner.invoke(
            commands,
            [
                "search",
                "--experiment-id",
                "1",
                "--extract-fields",
                "invalid.field",
                "--verbose",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid field path" in result.output
        assert "info.trace_id" in result.output
        assert "info.state" in result.output
        assert "info.request_time" in result.output
        assert "Tip: Use --verbose" not in result.output
