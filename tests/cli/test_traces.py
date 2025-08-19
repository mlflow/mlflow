import json
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli.traces import commands
from mlflow.entities import AssessmentSourceType

# Table cell formatting tests are in tests/utils/test_string_utils.py


@pytest.fixture
def runner():
    """Provide a CLI runner for testing."""
    return CliRunner()


def test_commands_group_exists():
    """Test that the traces command group is properly defined."""
    assert commands.name == "traces"
    assert commands.help is not None


def test_search_command_params():
    """Test that search command has all required parameters."""
    search_cmd = None
    for cmd in commands.commands.values():
        if cmd.name == "search":
            search_cmd = cmd
            break

    assert search_cmd is not None
    param_names = [p.name for p in search_cmd.params]

    # Check for required params
    assert "experiment_id" in param_names
    assert "filter_string" in param_names
    assert "max_results" in param_names
    assert "order_by" in param_names
    assert "page_token" in param_names
    assert "output" in param_names
    assert "extract_fields" in param_names


def test_get_command_params():
    """Test that get command has all required parameters."""
    get_cmd = None
    for cmd in commands.commands.values():
        if cmd.name == "get":
            get_cmd = cmd
            break

    assert get_cmd is not None
    param_names = [p.name for p in get_cmd.params]

    # Check for required params
    assert "trace_id" in param_names
    assert "extract_fields" in param_names


def test_assessment_source_type_choices():
    """Test that assessment commands use dynamic enum values."""
    log_feedback_cmd = None
    for cmd in commands.commands.values():
        if cmd.name == "log-feedback":
            log_feedback_cmd = cmd
            break

    assert log_feedback_cmd is not None

    # Find source_type param
    source_type_param = None
    for param in log_feedback_cmd.params:
        if param.name == "source_type":
            source_type_param = param
            break

    assert source_type_param is not None

    # Check that choices include the enum values
    assert AssessmentSourceType.HUMAN in source_type_param.type.choices
    assert AssessmentSourceType.LLM_JUDGE in source_type_param.type.choices
    assert AssessmentSourceType.CODE in source_type_param.type.choices


def test_search_command_with_fields(runner):
    """Test search command with field selection."""
    mock_trace = {
        "info": {
            "trace_id": "tr-123",
            "state": "OK",
            "request_time": 1700000000000,
            "execution_duration": 1234,
            "request_preview": "test request",
            "response_preview": "test response",
        }
    }

    # Create a mock trace object with a to_dict method
    mock_trace_obj = mock.Mock()
    mock_trace_obj.to_dict.return_value = mock_trace

    # Create a mock result that acts like a list but also has a token attribute
    mock_result = mock.Mock()
    mock_result.__iter__ = lambda self: iter([mock_trace_obj])
    mock_result.__getitem__ = lambda self, i: mock_trace_obj if i == 0 else None
    mock_result.__len__ = lambda self: 1
    mock_result.__bool__ = lambda self: True
    mock_result.token = None

    # Patch the TracingClient at the module level where it's imported
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_instance = mock.Mock()
        mock_client.return_value = mock_instance
        mock_instance.search_traces.return_value = mock_result

        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--extract-fields", "info.trace_id,info.state"],
        )

        assert result.exit_code == 0
        # Check that either the table output or the values appear in output
        assert "tr-123" in result.output
        assert "OK" in result.output


def test_get_command_with_fields(runner):
    """Test get command with field selection."""
    mock_trace = {
        "info": {"trace_id": "tr-123", "state": "OK"},
        "data": {"spans": [{"name": "test_span"}]},
    }

    # Create a mock trace object with a to_dict method
    mock_trace_obj = mock.Mock()
    mock_trace_obj.to_dict.return_value = mock_trace

    # Patch the TracingClient at the module level where it's imported
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_instance = mock.Mock()
        mock_client.return_value = mock_instance
        mock_instance.get_trace.return_value = mock_trace_obj

        result = runner.invoke(
            commands,
            ["get", "--trace-id", "tr-123", "--extract-fields", "info.trace_id"],
        )

        assert result.exit_code == 0

        # Parse JSON output
        output_json = json.loads(result.output)
        assert output_json == {"info": {"trace_id": "tr-123"}}


def test_delete_command(runner):
    """Test delete command."""
    # Patch the TracingClient at the module level where it's imported
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_instance = mock.Mock()
        mock_client.return_value = mock_instance
        mock_instance.delete_traces.return_value = 5

        result = runner.invoke(
            commands,
            ["delete", "--experiment-id", "1", "--trace-ids", "tr-1,tr-2,tr-3"],
        )

        assert result.exit_code == 0
        assert "Deleted 5 trace(s)" in result.output


def test_field_validation_error(runner):
    """Test that invalid fields produce helpful error messages."""
    mock_trace = {"info": {"trace_id": "tr-123"}}

    # Create a mock trace object with a to_dict method
    mock_trace_obj = mock.Mock()
    mock_trace_obj.to_dict.return_value = mock_trace

    # Create a mock result that acts like a list but also has a token attribute
    mock_result = mock.Mock()
    mock_result.__iter__ = lambda self: iter([mock_trace_obj])
    mock_result.__getitem__ = lambda self, i: mock_trace_obj if i == 0 else None
    mock_result.__len__ = lambda self: 1
    mock_result.__bool__ = lambda self: True
    mock_result.token = None

    # Patch the TracingClient at the module level where it's imported
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_instance = mock.Mock()
        mock_client.return_value = mock_instance
        mock_instance.search_traces.return_value = mock_result

        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--extract-fields", "invalid.field"],
        )

        assert result.exit_code != 0
        assert "Invalid field path" in result.output
        # Check for the tip about verbose mode
        assert "--verbose" in result.output


def test_field_validation_error_verbose_mode(runner):
    """Test that verbose mode shows all available fields."""
    mock_trace = {
        "info": {
            "trace_id": "tr-123",
            "state": "OK",
            "request_time": 1700000000000,
            "assessments": [{"id": "a1"}],
        },
        "data": {"spans": [{"name": "span1"}]},
    }

    # Create a mock trace object with a to_dict method
    mock_trace_obj = mock.Mock()
    mock_trace_obj.to_dict.return_value = mock_trace

    # Create a mock result that acts like a list but also has a token attribute
    mock_result = mock.Mock()
    mock_result.__iter__ = lambda self: iter([mock_trace_obj])
    mock_result.__getitem__ = lambda self, i: mock_trace_obj if i == 0 else None
    mock_result.__len__ = lambda self: 1
    mock_result.__bool__ = lambda self: True
    mock_result.token = None

    # Patch the TracingClient at the module level where it's imported
    with mock.patch("mlflow.cli.traces.TracingClient") as mock_client:
        mock_instance = mock.Mock()
        mock_client.return_value = mock_instance
        mock_instance.search_traces.return_value = mock_result

        # Use --verbose flag
        result = runner.invoke(
            commands,
            ["search", "--experiment-id", "1", "--extract-fields", "invalid.field", "--verbose"],
        )

        assert result.exit_code != 0
        assert "Invalid field path" in result.output
        # In verbose mode, we should see ALL the fields listed
        assert "info.trace_id" in result.output
        assert "info.state" in result.output
        assert "info.request_time" in result.output
        # Should NOT show the tip about --verbose since we're already using it
        assert "Tip: Use --verbose" not in result.output
