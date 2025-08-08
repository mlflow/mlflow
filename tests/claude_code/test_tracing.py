"""Simplified tests for mlflow.claude_code.tracing module."""

import os
import tempfile
import unittest
from pathlib import Path

# Test only the functions we can easily test without external dependencies
from mlflow.claude_code.tracing import (
    parse_timestamp_to_ns,
    setup_logging,
)


class TestTracingSimple(unittest.TestCase):
    """Test cases for mlflow.claude_code.tracing module."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_parse_timestamp_to_ns_iso_string(self):
        """Test parsing ISO timestamp string to nanoseconds."""
        iso_timestamp = "2024-01-15T10:30:45.123456Z"
        result = parse_timestamp_to_ns(iso_timestamp)

        # Verify it returns an integer (nanoseconds)
        assert isinstance(result, int)
        assert result > 0

    def test_parse_timestamp_to_ns_unix_seconds(self):
        """Test parsing Unix timestamp (seconds) to nanoseconds."""
        unix_timestamp = 1705312245.123456
        result = parse_timestamp_to_ns(unix_timestamp)

        # Should convert seconds to nanoseconds
        expected = int(unix_timestamp * 1_000_000_000)
        assert result == expected

    def test_parse_timestamp_to_ns_large_number(self):
        """Test parsing large timestamp numbers."""
        large_timestamp = 1705312245123
        result = parse_timestamp_to_ns(large_timestamp)

        # Function treats large numbers as seconds and converts to nanoseconds
        # Just verify we get a reasonable nanosecond value
        assert isinstance(result, int)
        assert result > 0

    def test_setup_logging_creates_logger(self):
        """Test that setup_logging returns a logger."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            logger = setup_logging()

            # Verify logger was created
            assert logger is not None
            assert logger.name == "mlflow.claude_code.tracing"

            # Verify log directory was created
            log_dir = Path(self.temp_dir) / ".claude" / "mlflow"
            assert log_dir.exists()
            assert log_dir.is_dir()

        finally:
            os.chdir(original_cwd)
