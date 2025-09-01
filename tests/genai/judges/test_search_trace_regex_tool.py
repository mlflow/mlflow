import json

from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.search_trace_regex import (
    SearchTraceRegexResult,
    SearchTraceRegexTool,
)


def create_test_trace():
    """Create a test trace with multiple spans and varied content."""
    trace_dict = {
        "info": {
            "request_id": "test-trace-123",
            "experiment_id": "0",
            "timestamp_ms": 1234567890,
            "execution_time_ms": 20,
            "status": "OK",
            "request_metadata": {
                "mlflow.trace_schema.version": "2",
            },
            "tags": {},
        },
        "data": {
            "spans": [
                {
                    "name": "user_authentication",
                    "context": {
                        "span_id": "0x1234567890abcdef",
                        "trace_id": "0xfedcba0987654321fedcba0987654321",
                    },
                    "parent_id": None,
                    "start_time": 1234567890000000000,
                    "end_time": 1234567900000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"test-trace-123"',
                        "mlflow.spanType": '"LLM"',
                        "mlflow.spanInputs": json.dumps(
                            {
                                "user_id": "12345",
                                "query": "What is the weather today?",
                                "context": "User is asking about weather information",
                            }
                        ),
                        "mlflow.spanOutputs": json.dumps(
                            {
                                "response": "I'll help you with the weather information.",
                                "status": "success",
                            }
                        ),
                        "model": "gpt-4",
                        "temperature": "0.7",
                        "system_message": "You are a helpful weather assistant",
                    },
                    "events": [],
                },
                {
                    "name": "weather_api_call",
                    "context": {
                        "span_id": "0xabcdef1234567890",
                        "trace_id": "0xfedcba0987654321fedcba0987654321",
                    },
                    "parent_id": "0x1234567890abcdef",
                    "start_time": 1234567901000000000,
                    "end_time": 1234567905000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"test-trace-123"',
                        "mlflow.spanType": '"TOOL"',
                        "mlflow.spanInputs": json.dumps(
                            {
                                "api_endpoint": "https://api.weather.com/current",
                                "location": "San Francisco, CA",
                                "api_key": "sk_test_key_12345",
                            }
                        ),
                        "mlflow.spanOutputs": json.dumps(
                            {
                                "weather_data": {
                                    "temperature": "22°C",
                                    "condition": "sunny",
                                    "humidity": "65%",
                                },
                                "error_code": None,
                            }
                        ),
                        "timeout": "30s",
                        "retry_count": "0",
                    },
                    "events": [],
                },
                {
                    "name": "error_handling",
                    "context": {
                        "span_id": "0x9876543210fedcba",
                        "trace_id": "0xfedcba0987654321fedcba0987654321",
                    },
                    "parent_id": "0x1234567890abcdef",
                    "start_time": 1234567906000000000,
                    "end_time": 1234567908000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"test-trace-123"',
                        "mlflow.spanType": '"UNKNOWN"',
                        "mlflow.spanInputs": json.dumps(
                            {
                                "error_message": "Connection timeout occurred",
                                "retry_attempt": 1,
                            }
                        ),
                        "mlflow.spanOutputs": json.dumps(
                            {
                                "handled": True,
                                "fallback_response": "Weather service temporarily unavailable",
                            }
                        ),
                        "error_type": "timeout",
                        "severity": "warning",
                    },
                    "events": [],
                },
            ],
            "request": '{"query": "weather"}',
            "response": '{"response": "Weather info"}',
        },
    }

    return Trace.from_dict(trace_dict)


def test_search_trace_regex_tool_name():
    tool = SearchTraceRegexTool()
    assert tool.name == "search_trace_regex"


def test_search_trace_regex_tool_definition():
    tool = SearchTraceRegexTool()
    definition = tool.get_definition()
    assert definition.type == "function"
    assert definition.function.name == "search_trace_regex"
    assert "regular expression" in definition.function.description.lower()
    assert "pattern" in definition.function.parameters.properties
    assert "max_matches" in definition.function.parameters.properties
    assert definition.function.parameters.required == ["pattern"]


def test_search_trace_regex_basic_search_success():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    result = tool.invoke(trace, pattern="weather")

    assert isinstance(result, SearchTraceRegexResult)
    assert result.pattern == "weather"
    assert result.error is None
    assert result.total_matches > 0
    assert len(result.matches) > 0

    # Should find weather-related matches
    weather_matches = [m for m in result.matches if "weather" in m.matched_text.lower()]
    assert len(weather_matches) > 0


def test_search_trace_regex_case_insensitive_search():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Search for "Weather" (capital W)
    result = tool.invoke(trace, pattern="Weather")

    assert result.total_matches > 0
    # Should find matches even though pattern has different case
    assert any("weather" in match.matched_text.lower() for match in result.matches)


def test_search_trace_regex_patterns():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Test regex pattern for user_id with digits
    result = tool.invoke(trace, pattern=r"user_id.*\d+")

    assert result.total_matches > 0
    assert any("user_id" in match.matched_text for match in result.matches)
    assert any("12345" in match.matched_text for match in result.matches)


def test_search_trace_regex_search_in_different_fields():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Search for something in inputs
    result_input = tool.invoke(trace, pattern="query.*weather")
    assert result_input.total_matches > 0

    # Search for something in outputs
    result_output = tool.invoke(trace, pattern="response.*help")
    assert result_output.total_matches > 0

    # Search for something in attributes
    result_attr = tool.invoke(trace, pattern="model.*gpt")
    assert result_attr.total_matches > 0


def test_search_trace_regex_surrounding_context():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    result = tool.invoke(trace, pattern="weather")

    # Check that matches include surrounding context
    for match in result.matches:
        assert len(match.surrounding_text) > len(match.matched_text)
        assert match.matched_text.lower() in match.surrounding_text.lower()


def test_search_trace_regex_max_matches_limit():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Use a pattern that should match many times
    result = tool.invoke(trace, pattern=".", max_matches=5)

    assert result.total_matches == 5
    assert len(result.matches) == 5


def test_search_trace_regex_no_matches():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    result = tool.invoke(trace, pattern="nonexistent_pattern_xyz")

    assert result.pattern == "nonexistent_pattern_xyz"
    assert result.total_matches == 0
    assert len(result.matches) == 0
    assert result.error is None


def test_search_trace_regex_invalid_regex():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    result = tool.invoke(trace, pattern="[invalid_regex")

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
    empty_trace = Trace(info=empty_trace_info, data=None)

    result = tool.invoke(empty_trace, pattern="test")

    assert result.total_matches == 0
    assert len(result.matches) == 0
    assert result.error == "Trace has no spans to search"


def test_search_trace_regex_trace_with_no_spans():
    tool = SearchTraceRegexTool()
    empty_data = TraceData(spans=[])
    empty_trace_info = TraceInfo(
        trace_id="no-spans-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=0,
    )
    no_spans_trace = Trace(info=empty_trace_info, data=empty_data)

    result = tool.invoke(no_spans_trace, pattern="test")

    assert result.total_matches == 0
    assert len(result.matches) == 0
    assert result.error == "Trace has no spans to search"


def test_search_trace_regex_span_id_in_matches():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    result = tool.invoke(trace, pattern="weather")

    # All matches should have valid span IDs (hex format from our test trace)
    valid_span_ids = {"1234567890abcdef", "abcdef1234567890", "9876543210fedcba"}
    for match in result.matches:
        assert match.span_id in valid_span_ids


def test_search_trace_regex_json_values_searchable():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Test that JSON values in outputs are searchable
    result = tool.invoke(trace, pattern="temperature.*22")

    assert result.total_matches > 0
    assert any("temperature" in match.matched_text for match in result.matches)


def test_search_trace_regex_complex_regex_patterns():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Test complex regex with word boundaries
    result = tool.invoke(trace, pattern=r"\bapi\b")

    assert result.total_matches > 0

    # Test regex with character classes
    result2 = tool.invoke(trace, pattern=r"[Tt]emperature")
    assert result2.total_matches > 0


def test_search_trace_regex_ellipses_in_surrounding_context():
    tool = SearchTraceRegexTool()
    # Create a trace with very long text to test ellipses
    long_text = "a" * 200 + "target_word" + "b" * 200

    long_trace_dict = {
        "info": {
            "request_id": "long-trace",
            "experiment_id": "0",
            "timestamp_ms": 1234567890,
            "execution_time_ms": 10,
            "status": "OK",
            "request_metadata": {
                "mlflow.trace_schema.version": "2",
            },
            "tags": {},
        },
        "data": {
            "spans": [
                {
                    "name": "test",
                    "context": {
                        "span_id": "0xfedcba0987654321",
                        "trace_id": "0x1234567890abcdef1234567890abcdef",
                    },
                    "parent_id": None,
                    "start_time": 1234567890000000000,
                    "end_time": 1234567900000000000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"long-trace"',
                        "mlflow.spanType": '"UNKNOWN"',
                        "mlflow.spanInputs": json.dumps({"long_input": long_text}),
                        "mlflow.spanOutputs": "{}",
                    },
                    "events": [],
                }
            ],
            "request": "{}",
            "response": "{}",
        },
    }

    long_trace = Trace.from_dict(long_trace_dict)

    result = tool.invoke(long_trace, pattern="target_word")

    assert result.total_matches >= 1
    match = result.matches[0]

    # Should have ellipses due to truncation
    assert match.surrounding_text.startswith("...")
    assert match.surrounding_text.endswith("...")
    assert "target_word" in match.surrounding_text


def test_search_trace_regex_default_max_matches():
    tool = SearchTraceRegexTool()
    trace = create_test_trace()
    # Test default value for max_matches parameter
    result = tool.invoke(trace, pattern=".")  # Should match many characters

    # Should use default limit (50)
    assert result.total_matches <= 50
