import json
import logging
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
    assert "all_pages" in param_names


def test_get_command_params():
    get_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "get"), None)
    assert get_cmd is not None
    param_names = [p.name for p in get_cmd.params]
    assert "trace_id" in param_names
    assert "extract_fields" in param_names


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


def test_search_command_strips_spaces_in_order_by(runner):
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = PagedList([], None)
        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--order-by", "timestamp_ms DESC, status"],
        )

    assert result.exit_code == 0
    _, kwargs = mock_client.return_value.search_traces.call_args
    assert kwargs["order_by"] == ["timestamp_ms DESC", "status"]


def _make_trace(trace_id: str) -> Trace:
    """Build a minimal Trace entity for search command tests."""
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=MlflowExperimentLocation(experiment_id="1"),
    )
    return Trace(
        info=TraceInfo(
            trace_id=trace_id,
            state=TraceState.OK,
            request_time=1700000000000,
            execution_duration=1234,
            request_preview="test request",
            response_preview="test response",
            trace_location=trace_location,
        ),
        data=TraceData(spans=[]),
    )


def test_search_command_jsonl_output(runner):
    traces = [_make_trace("tr-1"), _make_trace("tr-2")]

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = PagedList(traces, "next-token")
        result = runner.invoke(commands, ["search", "--experiment-id", "1", "--output", "jsonl"])

    assert result.exit_code == 0
    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    assert [line["info"]["trace_id"] for line in lines] == ["tr-1", "tr-2"]
    # Without --all-pages only the first page is emitted, even when a token is returned
    assert mock_client.return_value.search_traces.call_count == 1


def test_search_command_jsonl_output_with_extract_fields(runner):
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = PagedList([_make_trace("tr-1")], None)
        result = runner.invoke(
            commands,
            [
                "search",
                "--experiment-id",
                "1",
                "--output",
                "jsonl",
                "--extract-fields",
                "info.trace_id,info.state",
            ],
        )

    assert result.exit_code == 0
    assert json.loads(result.output.strip()) == {"info": {"trace_id": "tr-1", "state": "OK"}}


def test_search_command_all_pages_follows_page_token(runner):
    pages = [
        PagedList([_make_trace("tr-1")], "token-1"),
        PagedList([_make_trace("tr-2"), _make_trace("tr-3")], "token-2"),
        PagedList([_make_trace("tr-4")], None),
    ]

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.side_effect = pages
        result = runner.invoke(
            commands,
            [
                "search",
                "--experiment-id",
                "1",
                "--output",
                "jsonl",
                "--all-pages",
                "--max-results",
                "2",
            ],
        )

    assert result.exit_code == 0
    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    assert [line["info"]["trace_id"] for line in lines] == ["tr-1", "tr-2", "tr-3", "tr-4"]
    assert mock_client.return_value.search_traces.call_count == 3
    page_tokens = [
        call.kwargs["page_token"] for call in mock_client.return_value.search_traces.call_args_list
    ]
    assert page_tokens == [None, "token-1", "token-2"]


def test_search_command_all_pages_stops_on_empty_page(runner):
    pages = [
        PagedList([_make_trace("tr-1")], "token-1"),
        PagedList([], "token-2"),
    ]

    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.side_effect = pages
        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--output", "jsonl", "--all-pages"],
        )

    assert result.exit_code == 0
    assert len(result.output.strip().splitlines()) == 1
    assert mock_client.return_value.search_traces.call_count == 2


@pytest.mark.parametrize("output_format", ["table", "json"])
def test_search_command_all_pages_requires_jsonl_output(runner, output_format):
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.search_traces.return_value = PagedList([], None)
        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--output", output_format, "--all-pages"],
        )

    assert result.exit_code != 0
    assert "--all-pages requires --output jsonl" in result.output
    mock_client.return_value.search_traces.assert_not_called()


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


def test_delete_command(runner):
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.delete_traces.return_value = 5
        result = runner.invoke(
            commands,
            ["delete", "--experiment-id", "1", "--trace-ids", "tr-1,tr-2,tr-3"],
        )

        assert result.exit_code == 0
        assert "Deleted 5 trace(s)" in result.output


def test_delete_command_strips_spaces_in_trace_ids(runner):
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_client.return_value.delete_traces.return_value = 2
        result = runner.invoke(
            commands,
            ["delete", "--experiment-id", "1", "--trace-ids", "tr-1, tr-2"],
        )

    assert result.exit_code == 0
    _, kwargs = mock_client.return_value.delete_traces.call_args
    assert kwargs["trace_ids"] == ["tr-1", "tr-2"]


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
