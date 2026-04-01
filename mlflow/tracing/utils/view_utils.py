from __future__ import annotations

import json
import logging

from mlflow.entities.trace_view import SpanSelector, TraceView

try:
    from jsonpath_ng import parse as jsonpath_parse

    _HAS_JSONPATH = True
except ImportError:
    _HAS_JSONPATH = False

_logger = logging.getLogger(__name__)


def _unwrap_json_str(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _matches_span_type(span: dict, span_type: str) -> bool:
    if "span_type" in span and span["span_type"] == span_type:
        return True
    attrs = span.get("attributes", {})
    attr_type = attrs.get("mlflow.spanType")
    if attr_type is not None:
        return _unwrap_json_str(attr_type) == span_type
    return False


def _get_span_id(span: dict) -> str | None:
    ctx = span.get("context")
    if isinstance(ctx, dict):
        return ctx.get("span_id")
    return None


def _span_matches(span: dict, selector: SpanSelector) -> bool:
    if selector.span_name is not None and span.get("name") != selector.span_name:
        return False
    if selector.span_type is not None and not _matches_span_type(span, selector.span_type):
        return False
    if selector.span_id is not None and _get_span_id(span) != selector.span_id:
        return False
    if selector.attribute_key is not None:
        attrs = span.get("attributes", {})
        if selector.attribute_key not in attrs:
            return False
        if selector.attribute_value is not None:
            if str(attrs[selector.attribute_key]) != str(selector.attribute_value):
                return False
    return True


def _dfs_order(root: dict) -> list[dict]:
    result = []
    stack = [root]
    while stack:
        span = stack.pop()
        result.append(span)
        children = span.get("child_spans", [])
        stack.extend(reversed(children))
    return result


def _subtree_ids(span: dict) -> set[str]:
    ids = set()
    stack = [span]
    while stack:
        s = stack.pop()
        sid = _get_span_id(s)
        if sid:
            ids.add(sid)
        stack.extend(s.get("child_spans", []))
    return ids


def resolve_range(root: dict, span_range) -> list[dict]:
    spans_dfs = _dfs_order(root)

    from_idx = next(
        (i for i, s in enumerate(spans_dfs) if _span_matches(s, span_range.from_selector)),
        None,
    )
    if from_idx is None:
        return []

    from_span = spans_dfs[from_idx]

    if span_range.to_selector is None:
        sub_ids = _subtree_ids(from_span)
        return [s for s in spans_dfs[from_idx:] if _get_span_id(s) in sub_ids]

    to_idx = next(
        (j for j in range(from_idx + 1, len(spans_dfs))
         if _span_matches(spans_dfs[j], span_range.to_selector)),
        None,
    )
    if to_idx is None:
        sub_ids = _subtree_ids(from_span)
        return [s for s in spans_dfs[from_idx:] if _get_span_id(s) in sub_ids]

    to_span = spans_dfs[to_idx]

    to_sub_ids = _subtree_ids(to_span)
    end_idx = to_idx
    for k in range(to_idx + 1, len(spans_dfs)):
        if _get_span_id(spans_dfs[k]) in to_sub_ids:
            end_idx = k
        else:
            break

    return spans_dfs[from_idx:end_idx + 1]


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


def resolve_view(root: dict, view: TraceView) -> list[dict]:
    results = []
    for span_range in sorted(view.ranges, key=lambda r: r.position):
        matched = resolve_range(root, span_range)

        extracted_input = None
        extracted_output = None

        if matched and span_range.input_path:
            inp_data, _ = _extract_io(matched[0])
            if inp_data:
                extracted, success = apply_jsonpath(inp_data, span_range.input_path)
                if success:
                    extracted_input = extracted

            # Fallback: try extracting from outputs of first span
            if extracted_input is None:
                _, out_data = _extract_io(matched[0])
                if out_data:
                    extracted, success = apply_jsonpath(out_data, span_range.input_path)
                    if success:
                        extracted_input = extracted

        if matched and span_range.output_path:
            # Try each matched span from last to first for output extraction
            for span in reversed(matched):
                _, out_data = _extract_io(span)
                if out_data:
                    extracted, success = apply_jsonpath(out_data, span_range.output_path)
                    if success:
                        extracted_output = extracted
                        break

        results.append({
            "label": span_range.label,
            "description": span_range.description,
            "spans": matched,
            "extracted_input": extracted_input,
            "extracted_output": extracted_output,
        })
    return results
