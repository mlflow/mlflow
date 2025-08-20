import pytest

from mlflow.utils.jsonpath_utils import (
    filter_json_by_fields,
    jsonpath_extract_values,
    split_path_respecting_backticks,
    validate_field_paths,
)


def test_jsonpath_extract_values_simple():
    """Test simple field extraction."""
    data = {"info": {"trace_id": "tr-123", "state": "OK"}}
    values = jsonpath_extract_values(data, "info.trace_id")
    assert values == ["tr-123"]


def test_jsonpath_extract_values_nested():
    """Test nested field extraction."""
    data = {"info": {"metadata": {"user": "test@example.com"}}}
    values = jsonpath_extract_values(data, "info.metadata.user")
    assert values == ["test@example.com"]


def test_jsonpath_extract_values_wildcard_array():
    """Test wildcard extraction from arrays."""
    data = {"info": {"assessments": [{"feedback": {"value": 0.8}}, {"feedback": {"value": 0.9}}]}}
    values = jsonpath_extract_values(data, "info.assessments.*.feedback.value")
    assert values == [0.8, 0.9]


def test_jsonpath_extract_values_wildcard_dict():
    """Test wildcard extraction from dictionaries."""
    data = {"data": {"spans": {"span1": {"name": "first"}, "span2": {"name": "second"}}}}
    values = jsonpath_extract_values(data, "data.spans.*.name")
    assert set(values) == {"first", "second"}  # Order may vary with dict


def test_jsonpath_extract_values_missing_field():
    """Test extraction of missing fields."""
    data = {"info": {"trace_id": "tr-123"}}
    values = jsonpath_extract_values(data, "info.nonexistent")
    assert values == []


def test_jsonpath_extract_values_partial_path_missing():
    """Test extraction when part of path is missing."""
    data = {"info": {"trace_id": "tr-123"}}
    values = jsonpath_extract_values(data, "info.metadata.user")
    assert values == []


@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        ("info.trace_id", ["info", "trace_id"]),
        ("info.tags.`mlflow.traceName`", ["info", "tags", "mlflow.traceName"]),
        ("`field.one`.middle.`field.two`", ["field.one", "middle", "field.two"]),
        ("`mlflow.traceName`.value", ["mlflow.traceName", "value"]),
        ("info.`mlflow.traceName`", ["info", "mlflow.traceName"]),
    ],
)
def test_split_path_respecting_backticks(input_string, expected):
    """Test splitting paths with backtick-escaped segments."""
    assert split_path_respecting_backticks(input_string) == expected


def test_jsonpath_extract_values_with_backticks():
    """Test extraction with backtick-escaped field names containing dots."""
    # Field name with dot
    data = {"tags": {"mlflow.traceName": "test_trace"}}
    values = jsonpath_extract_values(data, "tags.`mlflow.traceName`")
    assert values == ["test_trace"]

    # Nested structure with dotted field names
    data = {"info": {"tags": {"mlflow.traceName": "my_trace", "user.id": "user123"}}}
    assert jsonpath_extract_values(data, "info.tags.`mlflow.traceName`") == ["my_trace"]
    assert jsonpath_extract_values(data, "info.tags.`user.id`") == ["user123"]

    # Mixed regular and backticked fields
    data = {"metadata": {"mlflow.source.type": "NOTEBOOK", "regular_field": "value"}}
    assert jsonpath_extract_values(data, "metadata.`mlflow.source.type`") == ["NOTEBOOK"]
    assert jsonpath_extract_values(data, "metadata.regular_field") == ["value"]


def test_jsonpath_extract_values_empty_array():
    """Test extraction from empty arrays."""
    data = {"info": {"assessments": []}}
    values = jsonpath_extract_values(data, "info.assessments.*.feedback.value")
    assert values == []


def test_jsonpath_extract_values_mixed_types():
    """Test extraction from mixed data types."""
    data = {
        "data": {
            "spans": [
                {"attributes": {"key1": "value1"}},
                {"attributes": {"key1": 42}},
                {"attributes": {"key1": True}},
            ]
        }
    }
    values = jsonpath_extract_values(data, "data.spans.*.attributes.key1")
    assert values == ["value1", 42, True]


def test_filter_json_by_fields_single_field():
    """Test filtering JSON by a single field."""
    data = {"info": {"trace_id": "tr-123", "state": "OK"}, "data": {"spans": []}}
    filtered = filter_json_by_fields(data, ["info.trace_id"])
    expected = {"info": {"trace_id": "tr-123"}}
    assert filtered == expected


def test_filter_json_by_fields_multiple_fields():
    """Test filtering JSON by multiple fields."""
    data = {
        "info": {"trace_id": "tr-123", "state": "OK", "unused": "value"},
        "data": {"spans": [], "metadata": {}},
    }
    filtered = filter_json_by_fields(data, ["info.trace_id", "info.state"])
    expected = {"info": {"trace_id": "tr-123", "state": "OK"}}
    assert filtered == expected


def test_filter_json_by_fields_wildcards():
    """Test filtering with wildcards preserves structure."""
    data = {
        "info": {
            "assessments": [
                {"feedback": {"value": 0.8}, "unused": "data"},
                {"feedback": {"value": 0.9}, "unused": "data"},
            ]
        }
    }
    filtered = filter_json_by_fields(data, ["info.assessments.*.feedback.value"])
    expected = {
        "info": {"assessments": [{"feedback": {"value": 0.8}}, {"feedback": {"value": 0.9}}]}
    }
    assert filtered == expected


def test_filter_json_by_fields_nested_arrays():
    """Test filtering nested arrays and objects."""
    data = {
        "data": {
            "spans": [
                {
                    "name": "span1",
                    "events": [
                        {"name": "event1", "data": "d1"},
                        {"name": "event2", "data": "d2"},
                    ],
                    "unused": "value",
                }
            ]
        }
    }
    filtered = filter_json_by_fields(data, ["data.spans.*.events.*.name"])
    expected = {"data": {"spans": [{"events": [{"name": "event1"}, {"name": "event2"}]}]}}
    assert filtered == expected


def test_filter_json_by_fields_missing_paths():
    """Test filtering with paths that don't exist."""
    data = {"info": {"trace_id": "tr-123"}}
    filtered = filter_json_by_fields(data, ["info.nonexistent", "missing.path"])
    assert filtered == {}


def test_filter_json_by_fields_partial_matches():
    """Test filtering with mix of existing and non-existing paths."""
    data = {"info": {"trace_id": "tr-123", "state": "OK"}}
    filtered = filter_json_by_fields(data, ["info.trace_id", "info.nonexistent"])
    expected = {"info": {"trace_id": "tr-123"}}
    assert filtered == expected


def test_validate_field_paths_valid():
    """Test validation of valid field paths."""
    data = {"info": {"trace_id": "tr-123", "assessments": [{"feedback": {"value": 0.8}}]}}
    # Should not raise any exception
    validate_field_paths(["info.trace_id", "info.assessments.*.feedback.value"], data)


def test_validate_field_paths_invalid():
    """Test validation of invalid field paths."""
    data = {"info": {"trace_id": "tr-123"}}

    with pytest.raises(ValueError, match="Invalid field path") as exc_info:
        validate_field_paths(["info.nonexistent"], data)

    assert "Invalid field path" in str(exc_info.value)
    assert "info.nonexistent" in str(exc_info.value)


def test_validate_field_paths_multiple_invalid():
    """Test validation with multiple invalid paths."""
    data = {"info": {"trace_id": "tr-123"}}

    with pytest.raises(ValueError, match="Invalid field path") as exc_info:
        validate_field_paths(["info.missing", "other.invalid"], data)

    error_msg = str(exc_info.value)
    assert "Invalid field path" in error_msg
    # Should mention both invalid paths
    assert "info.missing" in error_msg or "other.invalid" in error_msg


def test_validate_field_paths_suggestions():
    """Test that validation provides helpful suggestions."""
    data = {"info": {"trace_id": "tr-123", "assessments": [], "metadata": {}}}

    with pytest.raises(ValueError, match="Invalid field path") as exc_info:
        validate_field_paths(["info.traces"], data)  # Close to "trace_id"

    error_msg = str(exc_info.value)
    assert "Available fields" in error_msg
    assert "info.trace_id" in error_msg


def test_complex_trace_structure():
    """Test with realistic trace data structure."""
    trace_data = {
        "info": {
            "trace_id": "tr-abc123def",
            "state": "OK",
            "execution_duration": 1500,
            "assessments": [
                {
                    "assessment_id": "a-123",
                    "feedback": {"value": 0.85},
                    "source": {"source_type": "HUMAN", "source_id": "user@example.com"},
                }
            ],
            "tags": {"environment": "production", "mlflow.traceName": "test_trace"},
        },
        "data": {
            "spans": [
                {
                    "span_id": "span-1",
                    "name": "root_span",
                    "attributes": {"mlflow.spanType": "AGENT"},
                    "events": [{"name": "start", "attributes": {"key": "value"}}],
                }
            ]
        },
    }

    # Test various field extractions
    assert jsonpath_extract_values(trace_data, "info.trace_id") == ["tr-abc123def"]
    assert jsonpath_extract_values(trace_data, "info.assessments.*.feedback.value") == [0.85]
    assert jsonpath_extract_values(trace_data, "data.spans.*.name") == ["root_span"]
    assert jsonpath_extract_values(trace_data, "data.spans.*.events.*.name") == ["start"]

    # Test filtering preserves structure
    filtered = filter_json_by_fields(
        trace_data, ["info.trace_id", "info.assessments.*.feedback.value", "data.spans.*.name"]
    )

    assert "info" in filtered
    assert filtered["info"]["trace_id"] == "tr-abc123def"
    assert len(filtered["info"]["assessments"]) == 1
    assert filtered["info"]["assessments"][0]["feedback"]["value"] == 0.85
    assert "data" in filtered
    assert len(filtered["data"]["spans"]) == 1
    assert filtered["data"]["spans"][0]["name"] == "root_span"
    # Should not contain other fields
    assert "source" not in filtered["info"]["assessments"][0]
    assert "attributes" not in filtered["data"]["spans"][0]
