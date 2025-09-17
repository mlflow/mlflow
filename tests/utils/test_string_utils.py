import pytest

from mlflow.utils.string_utils import (
    format_table_cell_value,
    is_string_type,
    mslex_quote,
    strip_prefix,
    strip_suffix,
)


@pytest.mark.parametrize(
    (("original", "prefix", "expected")),
    [("smoketest", "smoke", "test"), ("", "test", ""), ("", "", ""), ("test", "", "test")],
)
def test_strip_prefix(original, prefix, expected):
    assert strip_prefix(original, prefix) == expected


@pytest.mark.parametrize(
    ("original", "suffix", "expected"),
    [("smoketest", "test", "smoke"), ("", "test", ""), ("", "", ""), ("test", "", "test")],
)
def test_strip_suffix(original, suffix, expected):
    assert strip_suffix(original, suffix) == expected


def test_is_string_type():
    assert is_string_type("validstring")
    assert is_string_type("")
    assert is_string_type((b"dog").decode("utf-8"))
    assert not is_string_type(None)
    assert not is_string_type(["teststring"])
    assert not is_string_type([])
    assert not is_string_type({})
    assert not is_string_type({"test": "string"})
    assert not is_string_type(12)
    assert not is_string_type(12.7)


def test_mslex_quote():
    assert mslex_quote("abc") == "abc"
    assert mslex_quote("a b c") == '"a b c"'
    assert mslex_quote("C:\\path\\to\\file") == "C:\\path\\to\\file"


def test_format_table_cell_value_basic():
    """Test basic cell value formatting."""
    result = format_table_cell_value("field", "value")
    assert result == "value"


def test_format_table_cell_value_empty():
    """Test formatting of empty values."""
    result = format_table_cell_value("field", None, [])
    assert result == "N/A"


def test_format_table_cell_value_multiple():
    """Test formatting of multiple values."""
    values = ["a", "b", "c", "d", "e"]
    result = format_table_cell_value("field", None, values)
    assert result == "a, b, c, ... (+2 more)"


def test_format_table_cell_value_timestamp():
    """Test timestamp formatting."""
    timestamp = "2025-01-15T10:31:24.123Z"
    result = format_table_cell_value("info.request_time", timestamp)
    assert "2025-01-15" in result
    assert "10:31:24" in result


def test_format_table_cell_value_duration_ms():
    """Test duration formatting in milliseconds."""
    result = format_table_cell_value("info.execution_duration_ms", 500)
    assert result == "500ms"


def test_format_table_cell_value_duration_seconds():
    """Test duration formatting in seconds."""
    result = format_table_cell_value("info.execution_duration_ms", 2500)
    assert result == "2.5s"


def test_format_table_cell_value_preview_truncation():
    """Test preview field truncation."""
    long_text = "This is a very long preview text that should be truncated"
    result = format_table_cell_value("info.request_preview", long_text)
    assert result == "This is a very lo..."
    assert len(result) == 20


def test_format_table_cell_value_invalid_timestamp():
    """Test handling of invalid timestamp values."""
    result = format_table_cell_value("info.request_time", "invalid-timestamp")
    assert result == "invalid-timestamp"  # Should return original


def test_format_table_cell_value_invalid_duration():
    """Test handling of invalid duration values."""
    result = format_table_cell_value("info.execution_duration_ms", "not-a-number")
    assert result == "not-a-number"  # Should return original
