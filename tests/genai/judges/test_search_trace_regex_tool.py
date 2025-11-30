import json

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.search_trace_regex import (
    SearchTraceRegexResult,
    SearchTraceRegexTool,
)


@pytest.fixture
def test_trace():
    """Create a test trace with varied content for testing."""
    trace_dict = {
        "info": {
            "request_id": "test-trace-123",
            "experiment_id": "0",
            "timestamp_ms": 1234567890,
            "execution_time_ms": 20,
            "status": "OK",
            "request_metadata": {"mlflow.trace_schema.version": "2"},
            "tags": {},
        },
        "data": {
            "spans": [
                {
                    "name": "weather_query",
                    "context": {"span_id": "0x123", "trace_id": "0xabc"},
                    "parent_id": None,
                    "start_time": 1234567890000000000,
                    "end_time": 1234567900000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"test-trace-123"',
                        "mlflow.spanInputs": json.dumps(
                            {"user_id": "12345", "query": "What is the weather today?"}
                        ),
                        "mlflow.spanOutputs": json.dumps(
                            {"response": "I'll help you with the weather information."}
                        ),
                        "model": "gpt-4",
                        "temperature": "22°C",
                    },
                    "events": [],
                }
            ],
            "request": '{"query": "weather"}',
            "response": '{"response": "Weather info"}',
        },
    }
    return Trace.from_dict(trace_dict)


def test_search_trace_regex_tool_metadata():
    tool = SearchTraceRegexTool()
    assert tool.name == "search_trace_regex"
    definition = tool.get_definition()
    assert definition.type == "function"
    assert definition.function.name == "search_trace_regex"
    assert "regular expression" in definition.function.description.lower()
    assert "pattern" in definition.function.parameters.properties
    assert "max_matches" in definition.function.parameters.properties
    assert "surrounding_content_length" in definition.function.parameters.properties
    assert definition.function.parameters.required == ["pattern"]


def test_search_trace_regex_basic_search_success(test_trace):
    tool = SearchTraceRegexTool()
    result = tool.invoke(test_trace, pattern="weather")

    assert isinstance(result, SearchTraceRegexResult)
    assert result.pattern == "weather"
    assert result.error is None
    assert result.total_matches > 0
    assert len(result.matches) > 0

    # Should find weather-related matches
    weather_matches = [m for m in result.matches if "weather" in m.matched_text.lower()]
    assert len(weather_matches) > 0


def test_search_trace_regex_case_insensitive_search(test_trace):
    tool = SearchTraceRegexTool()
    # Search for "Weather" (capital W)
    result = tool.invoke(test_trace, pattern="Weather")

    assert result.total_matches > 0
    # Should find matches even though pattern has different case
    assert any("weather" in match.matched_text.lower() for match in result.matches)


@pytest.mark.parametrize(
    ("pattern", "expected_content"),
    [
        (r"user_id.*\d+", ["user_id", "12345"]),
        ("query.*weather", ["query", "weather"]),
        ("response.*help", ["response", "help"]),
        ("model.*gpt", ["model", "gpt"]),
        (r"\bweather\b", ["weather"]),
        (r"[Tt]emperature", ["temperature"]),
    ],
)
def test_search_trace_regex_patterns(test_trace, pattern, expected_content):
    tool = SearchTraceRegexTool()
    result = tool.invoke(test_trace, pattern=pattern)
    assert result.total_matches > 0
    for content in expected_content:
        assert any(content.lower() in match.matched_text.lower() for match in result.matches)


def test_search_trace_regex_surrounding_context(test_trace):
    tool = SearchTraceRegexTool()
    result = tool.invoke(test_trace, pattern="weather")

    # Check that matches include surrounding context
    for match in result.matches:
        assert len(match.surrounding_text) > len(match.matched_text)
        assert match.matched_text.lower() in match.surrounding_text.lower()


def test_search_trace_regex_max_matches_limit(test_trace):
    tool = SearchTraceRegexTool()
    # Use a pattern that should match many times
    result = tool.invoke(test_trace, pattern=".", max_matches=5)

    assert result.total_matches == 5
    assert len(result.matches) == 5


def test_search_trace_regex_default_max_matches(test_trace):
    tool = SearchTraceRegexTool()
    # Test default value for max_matches parameter
    result = tool.invoke(test_trace, pattern=".")  # Should match many characters

    # Should use default limit (50)
    assert result.total_matches <= 50


def test_search_trace_regex_no_matches(test_trace):
    tool = SearchTraceRegexTool()
    result = tool.invoke(test_trace, pattern="nonexistent_pattern_xyz")

    assert result.pattern == "nonexistent_pattern_xyz"
    assert result.total_matches == 0
    assert len(result.matches) == 0
    assert result.error is None


def test_search_trace_regex_invalid_regex(test_trace):
    tool = SearchTraceRegexTool()
    result = tool.invoke(test_trace, pattern="[invalid_regex")

    assert result.pattern == "[invalid_regex"
    assert result.total_matches == 0
    assert len(result.matches) == 0
    assert result.error is not None
    assert "Invalid regex pattern" in result.error


def test_search_trace_regex_empty_trace():
    tool = SearchTraceRegexTool()
    empty_trace_info = TraceInfo(
        trace_id="empty-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=0,
    )
    empty_trace = Trace(info=empty_trace_info, data=TraceData(spans=[]))
    result = tool.invoke(empty_trace, pattern="empty-trace")
    assert result.total_matches > 0
    assert len(result.matches) > 0
    assert result.error is None


def test_search_trace_regex_span_id_in_matches(test_trace):
    tool = SearchTraceRegexTool()
    result = tool.invoke(test_trace, pattern="weather")

    # All matches should have the trace identifier
    for match in result.matches:
        assert match.span_id == "trace"


def test_search_trace_regex_json_values_searchable(test_trace):
    tool = SearchTraceRegexTool()
    # Test that JSON values in outputs are searchable
    result = tool.invoke(test_trace, pattern="temperature.*22")

    assert result.total_matches > 0
    assert any("temperature" in match.matched_text for match in result.matches)


def test_search_trace_regex_ellipses_in_surrounding_context():
    tool = SearchTraceRegexTool()
    long_text = "a" * 200 + "target_word" + "b" * 200
    trace_dict = {
        "info": {
            "request_id": "long-trace",
            "experiment_id": "0",
            "timestamp_ms": 1234567890,
            "execution_time_ms": 10,
            "status": "OK",
        },
        "data": {
            "spans": [
                {
                    "name": "test",
                    "context": {"span_id": "0x123", "trace_id": "0xabc"},
                    "parent_id": None,
                    "start_time": 1234567890000000000,
                    "end_time": 1234567900000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"long-trace"',
                        "mlflow.spanInputs": json.dumps({"long_input": long_text}),
                    },
                    "events": [],
                }
            ],
            "request": "{}",
            "response": "{}",
        },
    }
    trace = Trace.from_dict(trace_dict)
    result = tool.invoke(trace, pattern="target_word")
    assert result.total_matches >= 1
    match = result.matches[0]
    assert match.surrounding_text.startswith("...")
    assert match.surrounding_text.endswith("...")
    assert "target_word" in match.surrounding_text


def test_search_trace_regex_configurable_surrounding_content_length():
    """Test that the surrounding_content_length parameter controls the surrounding context size."""
    tool = SearchTraceRegexTool()
    # Create text with known positions: 50 'a's, then 'target', then 50 'b's
    long_text = "a" * 50 + "target" + "b" * 50
    trace_dict = {
        "info": {
            "request_id": "context-test",
            "experiment_id": "0",
            "timestamp_ms": 1234567890,
            "execution_time_ms": 10,
            "status": "OK",
        },
        "data": {
            "spans": [
                {
                    "name": "test",
                    "context": {"span_id": "0x123", "trace_id": "0xabc"},
                    "parent_id": None,
                    "start_time": 1234567890000000000,
                    "end_time": 1234567900000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"context-test"',
                        "mlflow.spanInputs": json.dumps({"input": long_text}),
                    },
                    "events": [],
                }
            ],
            "request": "{}",
            "response": "{}",
        },
    }
    trace = Trace.from_dict(trace_dict)

    # Test with small context window (10 characters)
    result_small = tool.invoke(trace, pattern="target", surrounding_content_length=10)
    assert result_small.total_matches >= 1
    match_small = result_small.matches[0]

    # Test with large context window (30 characters)
    result_large = tool.invoke(trace, pattern="target", surrounding_content_length=30)
    assert result_large.total_matches >= 1
    match_large = result_large.matches[0]

    # The large surrounding content length should include more surrounding text
    assert len(match_large.surrounding_text) > len(match_small.surrounding_text)
    assert "target" in match_small.surrounding_text
    assert "target" in match_large.surrounding_text


def test_search_trace_regex_default_surrounding_content_length(test_trace):
    """Test that the default surrounding_content_length parameter works correctly."""
    tool = SearchTraceRegexTool()

    # Test with explicit default value
    result_explicit = tool.invoke(test_trace, pattern="weather", surrounding_content_length=100)

    # Test with implicit default (should be same as explicit)
    result_implicit = tool.invoke(test_trace, pattern="weather")

    assert result_explicit.total_matches == result_implicit.total_matches
    assert len(result_explicit.matches) == len(result_implicit.matches)

    # The surrounding text should be the same length for both calls
    for match_exp, match_imp in zip(result_explicit.matches, result_implicit.matches):
        assert len(match_exp.surrounding_text) == len(match_imp.surrounding_text)
