"""Simplified tests for mlflow.claude_code.tracing module."""

# Test only the functions we can easily test without external dependencies
from mlflow.claude_code.tracing import (
    parse_timestamp_to_ns,
    setup_logging,
)


def test_parse_timestamp_to_ns_iso_string():
    """Test parsing ISO timestamp string to nanoseconds."""
    iso_timestamp = "2024-01-15T10:30:45.123456Z"
    result = parse_timestamp_to_ns(iso_timestamp)

    # Verify it returns an integer (nanoseconds)
    assert isinstance(result, int)
    assert result > 0


def test_parse_timestamp_to_ns_unix_seconds():
    """Test parsing Unix timestamp (seconds) to nanoseconds."""
    unix_timestamp = 1705312245.123456
    result = parse_timestamp_to_ns(unix_timestamp)

    # Should convert seconds to nanoseconds
    expected = int(unix_timestamp * 1_000_000_000)
    assert result == expected


def test_parse_timestamp_to_ns_large_number():
    """Test parsing large timestamp numbers."""
    large_timestamp = 1705312245123
    result = parse_timestamp_to_ns(large_timestamp)

    # Function treats large numbers as seconds and converts to nanoseconds
    # Just verify we get a reasonable nanosecond value
    assert isinstance(result, int)
    assert result > 0


def test_setup_logging_creates_logger(monkeypatch, tmp_path):
    """Test that setup_logging returns a logger."""
    monkeypatch.chdir(tmp_path)
    logger = setup_logging()

    # Verify logger was created
    assert logger is not None
    assert logger.name == "mlflow.claude_code.tracing"

    # Verify log directory was created
    log_dir = tmp_path / ".claude" / "mlflow"
    assert log_dir.exists()
    assert log_dir.is_dir()
