"""Tests for trace utility functions."""

import pytest

from mlflow.genai.judges.judge_trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
    extract_text_from_data,
)


def test_extract_text_from_data_string():
    """Test extracting text from string data."""
    result = extract_text_from_data("simple string", "request")
    assert result == "simple string"


def test_extract_text_from_data_dict_request():
    """Test extracting request text from dictionary data."""
    data = {"prompt": "test input", "other": "ignored"}
    result = extract_text_from_data(data, "request")
    assert result == "test input"


def test_extract_text_from_data_dict_response():
    """Test extracting response text from dictionary data."""
    data = {"content": "test output", "other": "ignored"}
    result = extract_text_from_data(data, "response")
    assert result == "test output"


def test_extract_text_from_data_none():
    """Test extracting text from None data."""
    result = extract_text_from_data(None, "request")
    assert result == ""


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
