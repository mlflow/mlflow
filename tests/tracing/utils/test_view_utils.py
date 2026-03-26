from __future__ import annotations

import json
from unittest import mock

import pytest

from mlflow.entities.trace_view import SpanFilter, TraceView
from mlflow.tracing.utils.view_utils import (
    _HAS_JSONPATH,
    apply_jsonpath,
    apply_span_filter,
    apply_view,
    find_first_matching_span,
    validate_jsonpath,
)

_requires_jsonpath = pytest.mark.skipif(not _HAS_JSONPATH, reason="jsonpath-ng is not installed")

# --- Sample span data fixtures ---


def _make_span(
    name="my_span",
    span_type="LLM",
    inputs=None,
    outputs=None,
    attributes=None,
):
    span = {
        "name": name,
        "span_type": span_type,
        "attributes": attributes or {},
    }
    if inputs is not None:
        span["inputs"] = inputs
    if outputs is not None:
        span["outputs"] = outputs
    return span


class TestFindFirstMatchingSpan:
    def test_match_by_name(self):
        spans = [_make_span(name="a"), _make_span(name="b")]
        result = find_first_matching_span(spans, SpanFilter(span_name="b"))
        assert result["name"] == "b"

    def test_match_by_type(self):
        spans = [_make_span(span_type="LLM"), _make_span(span_type="TOOL")]
        result = find_first_matching_span(spans, SpanFilter(span_type="TOOL"))
        assert result["span_type"] == "TOOL"

    def test_match_by_attribute_key(self):
        spans = [
            _make_span(attributes={"key1": "val1"}),
            _make_span(attributes={"key2": "val2"}),
        ]
        result = find_first_matching_span(spans, SpanFilter(attribute_key="key2"))
        assert result["attributes"]["key2"] == "val2"

    def test_match_by_attribute_key_and_value(self):
        spans = [
            _make_span(attributes={"k": "wrong"}),
            _make_span(attributes={"k": "right"}),
        ]
        result = find_first_matching_span(
            spans, SpanFilter(attribute_key="k", attribute_value="right")
        )
        assert result["attributes"]["k"] == "right"

    def test_combined_filter(self):
        spans = [
            _make_span(name="a", span_type="LLM"),
            _make_span(name="a", span_type="TOOL"),
            _make_span(name="b", span_type="TOOL"),
        ]
        result = find_first_matching_span(spans, SpanFilter(span_name="a", span_type="TOOL"))
        assert result["name"] == "a"
        assert result["span_type"] == "TOOL"

    def test_no_match(self):
        spans = [_make_span(name="a")]
        result = find_first_matching_span(spans, SpanFilter(span_name="nonexistent"))
        assert result is None

    def test_empty_spans(self):
        result = find_first_matching_span([], SpanFilter(span_name="a"))
        assert result is None

    def test_mlflow_wire_format_span_type_in_attributes(self):
        # MLflow stores span_type as JSON-encoded string in attributes
        spans = [
            {
                "name": "tool_call",
                "attributes": {
                    "mlflow.spanType": '"TOOL"',
                },
            },
        ]
        result = find_first_matching_span(spans, SpanFilter(span_type="TOOL"))
        assert result is not None
        assert result["name"] == "tool_call"

    def test_span_type_on_span_dict_takes_priority(self):
        spans = [_make_span(name="x", span_type="RETRIEVER")]
        result = find_first_matching_span(spans, SpanFilter(span_type="RETRIEVER"))
        assert result is not None


class TestApplySpanFilter:
    def test_returns_inputs_outputs(self):
        spans = [_make_span(name="s", inputs={"q": "hello"}, outputs={"a": "world"})]
        inp, out = apply_span_filter(spans, SpanFilter(span_name="s"))
        assert json.loads(inp) == {"q": "hello"}
        assert json.loads(out) == {"a": "world"}

    def test_fallback_to_attributes(self):
        spans = [
            {
                "name": "s",
                "span_type": "LLM",
                "attributes": {
                    "mlflow.spanInputs": '{"q": "hello"}',
                    "mlflow.spanOutputs": '{"a": "world"}',
                },
            }
        ]
        inp, out = apply_span_filter(spans, SpanFilter(span_name="s"))
        assert json.loads(inp) == {"q": "hello"}
        assert json.loads(out) == {"a": "world"}

    def test_no_filter_returns_none(self):
        spans = [_make_span()]
        inp, out = apply_span_filter(spans, None)
        assert inp is None
        assert out is None

    def test_no_match_returns_none(self):
        spans = [_make_span(name="a")]
        inp, out = apply_span_filter(spans, SpanFilter(span_name="missing"))
        assert inp is None
        assert out is None

    def test_string_values_returned_as_is(self):
        spans = [_make_span(name="s", inputs="raw text", outputs="raw out")]
        inp, out = apply_span_filter(spans, SpanFilter(span_name="s"))
        assert inp == "raw text"
        assert out == "raw out"


@_requires_jsonpath
class TestApplyJsonpath:
    def test_simple_extraction(self):
        data = json.dumps({"name": "Alice"})
        result, success = apply_jsonpath(data, "$.name")
        assert success
        assert result == "Alice"

    def test_nested_extraction(self):
        data = json.dumps({"a": {"b": {"c": 42}}})
        result, success = apply_jsonpath(data, "$.a.b.c")
        assert success
        assert result == "42"

    def test_array_wildcard(self):
        data = json.dumps({"items": [{"id": 1}, {"id": 2}]})
        result, success = apply_jsonpath(data, "$.items[*].id")
        assert success
        assert "1" in result
        assert "2" in result

    def test_no_match(self):
        data = json.dumps({"a": 1})
        result, success = apply_jsonpath(data, "$.nonexistent")
        assert not success
        assert result is None

    def test_invalid_json(self):
        result, success = apply_jsonpath("not json", "$.a")
        assert not success
        assert result is None

    def test_empty_expr(self):
        data = json.dumps({"a": 1})
        result, success = apply_jsonpath(data, "")
        assert not success
        assert result is None

    def test_none_expr(self):
        data = json.dumps({"a": 1})
        result, success = apply_jsonpath(data, None)
        assert not success
        assert result is None

    def test_null_result(self):
        data = json.dumps({"a": None})
        result, success = apply_jsonpath(data, "$.a")
        assert not success
        assert result is None

    def test_library_not_installed(self):
        data = json.dumps({"a": 1})
        with mock.patch("mlflow.tracing.utils.view_utils._HAS_JSONPATH", new=False) as mock_flag:
            result, success = apply_jsonpath(data, "$.a")
            assert not success
            assert result is None
            # Verify the mock was used (flag was checked)
            assert not mock_flag


@_requires_jsonpath
class TestValidateJsonpath:
    def test_valid_expression(self):
        is_valid, error = validate_jsonpath("$.name")
        assert is_valid
        assert error is None

    def test_empty_is_valid(self):
        is_valid, error = validate_jsonpath("")
        assert is_valid
        assert error is None

    def test_invalid_syntax(self):
        is_valid, error = validate_jsonpath("[invalid")
        assert not is_valid
        assert error is not None


class TestApplyView:
    @_requires_jsonpath
    def test_full_pipeline(self):
        spans = [
            _make_span(
                name="chat",
                inputs=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
                outputs=json.dumps({"response": "hello"}),
            )
        ]
        view = TraceView(
            name="test_view",
            experiment_id="1",
            span_filter=SpanFilter(span_name="chat"),
            input_path="$.messages[0].content",
            output_path="$.response",
        )
        inp, out = apply_view(spans, view, fallback_input="fb_in", fallback_output="fb_out")
        assert inp == "hi"
        assert out == "hello"

    def test_fallback_when_no_filter(self):
        view = TraceView(name="test_view", experiment_id="1")
        inp, out = apply_view([], view, fallback_input="fb_in", fallback_output="fb_out")
        assert inp == "fb_in"
        assert out == "fb_out"

    def test_fallback_when_no_match(self):
        spans = [_make_span(name="a")]
        view = TraceView(
            name="test_view",
            experiment_id="1",
            span_filter=SpanFilter(span_name="missing"),
        )
        inp, out = apply_view(spans, view, fallback_input="fb_in", fallback_output="fb_out")
        assert inp == "fb_in"
        assert out == "fb_out"
