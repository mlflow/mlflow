"""Simplified tests for mlflow.claude_code.cli module."""

import unittest

from click.testing import CliRunner

from mlflow.claude_code.cli import commands


class TestCLISimple(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_claude_help_command(self):
        """Test that the main claude command shows help."""
        result = self.runner.invoke(commands, ["--help"])
        assert result.exit_code == 0
        assert "Commands for Claude Code integration" in result.output
        assert "trace" in result.output

    def test_trace_command_help(self):
        """Test that the trace command shows help."""
        result = self.runner.invoke(commands, ["trace", "--help"])
        assert result.exit_code == 0
        assert "Set up Claude Code tracing" in result.output
        assert "--tracking-uri" in result.output
        assert "--experiment-id" in result.output
        assert "--disable" in result.output
        assert "--status" in result.output

    def test_trace_status_with_no_config(self):
        """Test trace status when no config exists."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(commands, ["trace", "--status"])
            assert result.exit_code == 0
            assert "❌ Claude tracing is not enabled" in result.output

    def test_trace_disable_with_no_config(self):
        """Test trace disable when no config exists."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(commands, ["trace", "--disable"])
            assert result.exit_code == 0
            # Should handle gracefully even if no config exists
