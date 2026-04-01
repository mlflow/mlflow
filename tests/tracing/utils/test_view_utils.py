from __future__ import annotations

import json

import pytest

from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.tracing.utils.view_utils import (
    apply_jsonpath,
    resolve_range,
    resolve_view,
    validate_jsonpath,
)


def _make_span(name, span_type, span_id, inputs=None, outputs=None, children=None):
    return {
        "name": name,
        "span_type": span_type,
        "context": {"span_id": span_id},
        "inputs": json.dumps(inputs or {}),
        "outputs": json.dumps(outputs or {}),
        "attributes": {},
        "child_spans": children or [],
    }


def _build_trace():
    """Build a tree matching the prototype demo trace."""
    tokenizer = _make_span("tokenizer", "INTERNAL", "s2a", outputs={"tokens": 150})
    llm1 = _make_span(
        "ChatOpenAI", "LLM", "s2",
        outputs={"reasoning": "Need to search", "tool_call": "search"},
        children=[tokenizer],
    )
    sql = _make_span("sql_query", "INTERNAL", "s3a", inputs={"sql": "SELECT *"}, outputs={"rows": 8})
    tool1 = _make_span(
        "search_content", "TOOL", "s3",
        inputs={"query": "template"},
        outputs={"results": [{"title": "Welcome"}], "count": 8},
        children=[sql],
    )
    llm2 = _make_span("ChatOpenAI", "LLM", "s4", outputs={"reasoning": "Found template"})
    tool2 = _make_span(
        "get_transaction", "TOOL", "s5",
        outputs={"transaction": {"buyer": "John"}},
    )
    llm3 = _make_span("ChatOpenAI", "LLM", "s6", outputs={"reasoning": "Need participants"})
    tool3 = _make_span("list_participants", "TOOL", "s7", outputs={"participants": []})
    llm4 = _make_span(
        "ChatOpenAI", "LLM", "s8",
        outputs={"reasoning": "Must escalate", "decision": "escalate"},
    )
    root = _make_span(
        "AgentExecutor", "CHAIN", "s1",
        inputs={"task": "send email"},
        outputs={"status": "escalated"},
        children=[llm1, tool1, llm2, tool2, llm3, tool3, llm4],
    )
    return root


class TestResolveRange:
    def test_single_span_by_id(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_id="s8"))
        matched = resolve_range(root, r)
        assert len(matched) == 1
        assert matched[0]["context"]["span_id"] == "s8"

    def test_span_with_subtree(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_id="s2"))
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s2", "s2a"]

    def test_range_between_two_spans(self):
        root = _build_trace()
        r = SpanRange(
            from_selector=SpanSelector(span_id="s2"),
            to_selector=SpanSelector(span_name="search_content"),
        )
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s2", "s2a", "s3", "s3a"]

    def test_range_by_type(self):
        root = _build_trace()
        r = SpanRange(
            from_selector=SpanSelector(span_id="s4"),
            to_selector=SpanSelector(span_name="get_transaction"),
        )
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s4", "s5"]

    def test_root_span_matches_all(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_name="AgentExecutor"))
        matched = resolve_range(root, r)
        assert len(matched) == 10

    def test_no_match_returns_empty(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_name="nonexistent"))
        matched = resolve_range(root, r)
        assert matched == []

    def test_to_selector_not_found_falls_back_to_subtree(self):
        root = _build_trace()
        r = SpanRange(
            from_selector=SpanSelector(span_id="s2"),
            to_selector=SpanSelector(span_name="nonexistent"),
        )
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s2", "s2a"]


class TestResolveView:
    def test_full_view_resolution(self):
        root = _build_trace()
        view = TraceView(
            name="test",
            trace_id="t1",
            ranges=[
                SpanRange(
                    from_selector=SpanSelector(span_name="AgentExecutor"),
                    label="Summary",
                    description="Overview",
                    position=0,
                ),
                SpanRange(
                    from_selector=SpanSelector(span_id="s2"),
                    to_selector=SpanSelector(span_name="search_content"),
                    label="Template Lookup",
                    description="Searched for template",
                    input_path="$.reasoning",
                    output_path="$.results",
                    position=1,
                ),
                SpanRange(
                    from_selector=SpanSelector(span_id="s8"),
                    label="Escalation",
                    description="Agent escalated",
                    output_path="$.reasoning",
                    position=2,
                ),
            ],
        )
        results = resolve_view(root, view)
        assert len(results) == 3

        assert results[0]["label"] == "Summary"
        assert len(results[0]["spans"]) == 10

        assert results[1]["label"] == "Template Lookup"
        assert len(results[1]["spans"]) == 4
        assert results[1]["extracted_input"] == "Need to search"
        assert results[1]["extracted_output"] == "[{'title': 'Welcome'}]"

        assert results[2]["label"] == "Escalation"
        assert len(results[2]["spans"]) == 1
        assert results[2]["extracted_output"] == "Must escalate"


class TestApplyJsonpath:
    def test_simple_path(self):
        data = json.dumps({"reasoning": "hello"})
        result, success = apply_jsonpath(data, "$.reasoning")
        assert success
        assert result == "hello"

    def test_no_match(self):
        data = json.dumps({"other": "value"})
        result, success = apply_jsonpath(data, "$.nonexistent")
        assert not success

    def test_none_expr(self):
        result, success = apply_jsonpath('{"a": 1}', None)
        assert not success


class TestValidateJsonpath:
    def test_valid(self):
        ok, err = validate_jsonpath("$.field")
        assert ok
        assert err is None

    def test_empty_is_valid(self):
        ok, err = validate_jsonpath("")
        assert ok

    def test_invalid(self):
        ok, err = validate_jsonpath("[invalid")
        assert not ok
        assert err is not None
