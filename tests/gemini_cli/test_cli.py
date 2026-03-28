import pytest
from click.testing import CliRunner

from mlflow.gemini_cli.cli import gemini_cli


@pytest.fixture
def runner():
    """Provide a CLI runner for tests."""
    return CliRunner()


def test_gemini_cli_help_command(runner):
    result = runner.invoke(gemini_cli, ["--help"])
    assert result.exit_code == 0
    assert "Set up Gemini CLI tracing" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_gemini_cli_status_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(gemini_cli, ["--status"])
        assert result.exit_code == 0
        assert "Gemini CLI tracing is not enabled" in result.output


def test_gemini_cli_disable_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(gemini_cli, ["--disable"])
        assert result.exit_code == 0
        assert "No Gemini CLI configuration found" in result.output


def test_gemini_cli_setup_creates_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(gemini_cli, ["."])
        assert result.exit_code == 0
        assert "Gemini CLI hooks configured" in result.output
        assert "Gemini CLI Tracing Setup Complete!" in result.output


def test_gemini_cli_setup_with_tracking_uri(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(gemini_cli, [".", "-u", "http://localhost:5000"])
        assert result.exit_code == 0
        assert "http://localhost:5000" in result.output


def test_gemini_cli_setup_with_experiment_id(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(gemini_cli, [".", "-e", "123456"])
        assert result.exit_code == 0
        assert "123456" in result.output
