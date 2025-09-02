"""Tests for trace utility functions."""

from unittest.mock import Mock

import pytest

from mlflow.entities.trace import Trace
from mlflow.genai.judges.judge_trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
)


@pytest.mark.parametrize(
    ("trace_fixture", "expected_content"),
    [
        ("sample_trace_with_assessment", "test input"),
        ("trace_with_nested_request_response", "nested input"),
        ("trace_with_list_request_response", "item"),
        ("trace_with_string_request_response", "capital of France"),
        ("trace_with_mixed_types", "test"),
    ],
)
def test_extract_request_from_trace(trace_fixture, expected_content, request):
    """Test extracting request from various trace types."""
    trace = request.getfixturevalue(trace_fixture)
    result = extract_request_from_trace(trace)
    assert expected_content in result


@pytest.mark.parametrize(
    ("trace_fixture", "expected_content"),
    [
        ("sample_trace_with_assessment", "test output"),
        ("trace_with_nested_request_response", "nested output"),
        ("trace_with_list_request_response", "result"),
        ("trace_with_string_request_response", "Paris"),
        ("trace_with_mixed_types", "response"),
    ],
)
def test_extract_response_from_trace(trace_fixture, expected_content, request):
    """Test extracting response from various trace types."""
    trace = request.getfixturevalue(trace_fixture)
    result = extract_response_from_trace(trace)
    assert expected_content in result


def test_extract_request_from_trace_no_data():
    """Test extracting request when trace has no data attribute."""
    mock_trace = Mock(spec=Trace)
    del mock_trace.data  # Remove data attribute
    result = extract_request_from_trace(mock_trace)
    assert result == ""


def test_extract_request_from_trace_none_data():
    """Test extracting request when trace.data is None."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = None
    result = extract_request_from_trace(mock_trace)
    assert result == ""


def test_extract_request_from_trace_no_spans():
    """Test extracting request when trace.data has no spans attribute."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    del mock_trace.data.spans  # Remove spans attribute
    result = extract_request_from_trace(mock_trace)
    assert result == ""


def test_extract_request_from_trace_empty_spans():
    """Test extracting request when trace.data.spans is empty."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_trace.data.spans = []
    result = extract_request_from_trace(mock_trace)
    assert result == ""


def test_extract_request_from_trace_no_inputs():
    """Test extracting request when first span has no inputs attribute."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_span = Mock()
    del mock_span.inputs  # Remove inputs attribute
    mock_trace.data.spans = [mock_span]
    result = extract_request_from_trace(mock_trace)
    assert result == ""


def test_extract_request_from_trace_none_inputs():
    """Test extracting request when first span.inputs is None."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_span = Mock()
    mock_span.inputs = None
    mock_trace.data.spans = [mock_span]
    result = extract_request_from_trace(mock_trace)
    assert result == ""


def test_extract_response_from_trace_no_data():
    """Test extracting response when trace has no data attribute."""
    mock_trace = Mock(spec=Trace)
    del mock_trace.data  # Remove data attribute
    result = extract_response_from_trace(mock_trace)
    assert result == ""


def test_extract_response_from_trace_none_data():
    """Test extracting response when trace.data is None."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = None
    result = extract_response_from_trace(mock_trace)
    assert result == ""


def test_extract_response_from_trace_no_spans():
    """Test extracting response when trace.data has no spans attribute."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    del mock_trace.data.spans  # Remove spans attribute
    result = extract_response_from_trace(mock_trace)
    assert result == ""


def test_extract_response_from_trace_empty_spans():
    """Test extracting response when trace.data.spans is empty."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_trace.data.spans = []
    result = extract_response_from_trace(mock_trace)
    assert result == ""


def test_extract_response_from_trace_no_outputs():
    """Test extracting response when first span has no outputs attribute."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_span = Mock()
    del mock_span.outputs  # Remove outputs attribute
    mock_trace.data.spans = [mock_span]
    result = extract_response_from_trace(mock_trace)
    assert result == ""


def test_extract_response_from_trace_none_outputs():
    """Test extracting response when first span.outputs is None."""
    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_span = Mock()
    mock_span.outputs = None
    mock_trace.data.spans = [mock_span]
    result = extract_response_from_trace(mock_trace)
    assert result == ""
