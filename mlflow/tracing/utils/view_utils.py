from __future__ import annotations

import json
import logging

from mlflow.entities.trace_view import SpanFilter, TraceView

try:
    from jsonpath_ng import parse as jsonpath_parse

    _HAS_JSONPATH = True
except ImportError:
    _HAS_JSONPATH = False

_logger = logging.getLogger(__name__)


def _unwrap_json_str(value):
    """If value is a JSON-encoded string, decode it once."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _matches_span_type(span: dict, span_type: str) -> bool:
    # Check direct span_type field first
    if "span_type" in span and span["span_type"] == span_type:
        return True
    # Fallback: check attributes with JSON unwrapping
    attrs = span.get("attributes", {})
    attr_type = attrs.get("mlflow.spanType")
    if attr_type is not None:
        return _unwrap_json_str(attr_type) == span_type
    return False


def find_first_matching_span(spans: list[dict], filter_config: SpanFilter) -> dict | None:
    return next(
        (span for span in spans if _span_matches(span, filter_config)),
        None,
    )


def _span_matches(span: dict, f: SpanFilter) -> bool:
    if f.span_name is not None and span.get("name") != f.span_name:
        return False
    if f.span_type is not None and not _matches_span_type(span, f.span_type):
        return False
    if f.attribute_key is not None:
        attrs = span.get("attributes", {})
        if f.attribute_key not in attrs:
            return False
        if f.attribute_value is not None:
            if str(attrs[f.attribute_key]) != str(f.attribute_value):
                return False
    return True


def _extract_io(span: dict) -> tuple[str | None, str | None]:
    inputs = span.get("inputs")
    if inputs is None:
        raw = span.get("attributes", {}).get("mlflow.spanInputs")
        if raw is not None:
            inputs = _unwrap_json_str(raw)

    outputs = span.get("outputs")
    if outputs is None:
        raw = span.get("attributes", {}).get("mlflow.spanOutputs")
        if raw is not None:
            outputs = _unwrap_json_str(raw)

    def _serialize(val):
        if val is None:
            return None
        if isinstance(val, str):
            return val
        return json.dumps(val)

    return _serialize(inputs), _serialize(outputs)


def apply_span_filter(
    spans_data: list[dict], filter_config: SpanFilter | None
) -> tuple[str | None, str | None]:
    if filter_config is None:
        return None, None
    span = find_first_matching_span(spans_data, filter_config)
    if span is None:
        return None, None
    return _extract_io(span)


def apply_jsonpath(data: str, jsonpath_expr: str | None) -> tuple[str | None, bool]:
    if not jsonpath_expr:
        return None, False
    if not _HAS_JSONPATH:
        _logger.warning("jsonpath-ng is not installed. Install it with: pip install jsonpath-ng")
        return None, False
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, False

    try:
        expr = jsonpath_parse(jsonpath_expr)
    except Exception:
        return None, False

    matches = expr.find(parsed)
    if not matches:
        return None, False

    values = [m.value for m in matches if m.value is not None]
    if not values:
        return None, False

    parts = [str(v) for v in values]
    result = "\n".join(parts)
    return result, True


def validate_jsonpath(expr: str) -> tuple[bool, str | None]:
    if not expr:
        return True, None
    if not _HAS_JSONPATH:
        return False, "jsonpath-ng is not installed"
    try:
        jsonpath_parse(expr)
        return True, None
    except Exception as e:
        return False, str(e)


def apply_view(
    trace_spans_data: list[dict],
    view: TraceView,
    fallback_input: str | None,
    fallback_output: str | None,
) -> tuple[str, str]:
    inp, out = apply_span_filter(trace_spans_data, view.span_filter)

    if inp is not None and view.input_path:
        extracted, success = apply_jsonpath(inp, view.input_path)
        if success:
            inp = extracted

    if out is not None and view.output_path:
        extracted, success = apply_jsonpath(out, view.output_path)
        if success:
            out = extracted

    final_input = inp if inp is not None else (fallback_input or "")
    final_output = out if out is not None else (fallback_output or "")
    return final_input, final_output
