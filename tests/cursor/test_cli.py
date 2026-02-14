import pytest
from click.testing import CliRunner

from mlflow.cursor.cli import commands


@pytest.fixture
def runner():
    return CliRunner()


def test_cursor_help_command(runner):
    result = runner.invoke(commands, ["--help"])
    assert result.exit_code == 0
    assert "Commands for autologging with MLflow" in result.output
    assert "cursor" in result.output


def test_trace_command_help(runner):
    result = runner.invoke(commands, ["cursor", "--help"])
    assert result.exit_code == 0
    assert "Set up Cursor tracing" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_trace_status_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["cursor", "--status"])
        assert result.exit_code == 0
        assert "Cursor tracing is not enabled" in result.output


def test_trace_disable_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["cursor", "--disable"])
        assert result.exit_code == 0
        # Should handle gracefully even if no config exists
