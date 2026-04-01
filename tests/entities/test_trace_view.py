from __future__ import annotations

import json

import pytest

from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.exceptions import MlflowException


class TestSpanSelector:
    def test_creation_defaults(self):
        s = SpanSelector()
        assert s.span_name is None
        assert s.span_type is None
        assert s.span_id is None
        assert s.attribute_key is None
        assert s.attribute_value is None

    def test_creation_with_values(self):
        s = SpanSelector(
            span_name="my_span",
            span_type="LLM",
            span_id="span-1",
            attribute_key="model",
            attribute_value="gpt-4",
        )
        assert s.span_name == "my_span"
        assert s.span_type == "LLM"
        assert s.span_id == "span-1"

    def test_to_dict(self):
        s = SpanSelector(span_name="my_span", span_type="LLM")
        d = s.to_dict()
        assert d == {
            "span_name": "my_span",
            "span_type": "LLM",
            "span_id": None,
            "attribute_key": None,
            "attribute_value": None,
        }

    def test_from_dict(self):
        d = {"span_name": "x", "span_type": "CHAIN", "span_id": "s1"}
        s = SpanSelector.from_dict(d)
        assert s.span_name == "x"
        assert s.span_type == "CHAIN"
        assert s.span_id == "s1"

    def test_from_dict_partial(self):
        s = SpanSelector.from_dict({"span_name": "only_name"})
        assert s.span_name == "only_name"
        assert s.span_type is None
        assert s.span_id is None

    def test_json_round_trip(self):
        s = SpanSelector(span_name="test", span_id="s1", attribute_key="key")
        json_str = s.to_json()
        restored = SpanSelector.from_json(json_str)
        assert restored == s

    def test_to_json_is_valid_json(self):
        s = SpanSelector(span_name="test")
        parsed = json.loads(s.to_json())
        assert isinstance(parsed, dict)


class TestSpanRange:
    def test_creation_minimal(self):
        r = SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            label="LLM Call",
            description="An LLM call",
        )
        assert r.from_selector.span_type == "LLM"
        assert r.to_selector is None
        assert r.label == "LLM Call"
        assert r.input_path is None
        assert r.output_path is None
        assert r.position == 0

    def test_creation_full(self):
        r = SpanRange(
            from_selector=SpanSelector(span_id="s1"),
            to_selector=SpanSelector(span_name="search"),
            label="Template Lookup",
            description="Searched for template",
            input_path="$.reasoning",
            output_path="$.results",
            position=1,
            range_id="r-abc",
        )
        assert r.to_selector.span_name == "search"
        assert r.input_path == "$.reasoning"
        assert r.range_id == "r-abc"
        assert r.position == 1

    def test_to_dict_round_trip(self):
        r = SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            to_selector=SpanSelector(span_name="tool"),
            label="Step 1",
            description="First step",
            input_path="$.input",
            output_path="$.output",
            position=0,
            range_id="r-1",
        )
        d = r.to_dict()
        restored = SpanRange.from_dict(d)
        assert restored.from_selector == r.from_selector
        assert restored.to_selector == r.to_selector
        assert restored.label == r.label
        assert restored.description == r.description
        assert restored.input_path == r.input_path
        assert restored.output_path == r.output_path
        assert restored.position == r.position
        assert restored.range_id == r.range_id

    def test_to_dict_without_to_selector(self):
        r = SpanRange(
            from_selector=SpanSelector(span_id="s1"),
            label="Solo",
            description="Single span",
        )
        d = r.to_dict()
        assert d["to_selector"] is None
        restored = SpanRange.from_dict(d)
        assert restored.to_selector is None


class TestTraceView:
    def test_trace_scoped_creation(self):
        tv = TraceView(name="my_view", trace_id="tr-123")
        assert tv.name == "my_view"
        assert tv.trace_id == "tr-123"
        assert tv.experiment_id is None
        assert tv.ranges == []

    def test_experiment_scoped_creation(self):
        tv = TraceView(name="exp_view", experiment_id="exp-456")
        assert tv.experiment_id == "exp-456"
        assert tv.trace_id is None

    def test_scope_property(self):
        assert TraceView(name="v", trace_id="t1").scope == "trace"
        assert TraceView(name="v", experiment_id="e1").scope == "experiment"

    def test_validate_scope_both_set_raises(self):
        tv = TraceView(name="v", trace_id="t1", experiment_id="e1")
        with pytest.raises(MlflowException, match="Exactly one of"):
            tv.validate_scope()

    def test_validate_scope_neither_set_raises(self):
        tv = TraceView(name="v")
        with pytest.raises(MlflowException, match="Exactly one of"):
            tv.validate_scope()

    def test_validate_scope_valid(self):
        TraceView(name="v", trace_id="t1").validate_scope()
        TraceView(name="v", experiment_id="e1").validate_scope()

    def test_to_dict_round_trip_with_ranges(self):
        tv = TraceView(
            name="view1",
            trace_id="tr-1",
            ranges=[
                SpanRange(
                    from_selector=SpanSelector(span_name="root"),
                    label="Summary",
                    description="A summary",
                    position=0,
                    range_id="r-1",
                ),
                SpanRange(
                    from_selector=SpanSelector(span_type="LLM"),
                    to_selector=SpanSelector(span_type="TOOL"),
                    label="Step 1",
                    description="First step",
                    input_path="$.reasoning",
                    position=1,
                    range_id="r-2",
                ),
            ],
            created_by="assistant",
            view_id="vid-1",
            create_time_ms=1000,
            last_update_time_ms=2000,
        )
        d = tv.to_dict()
        restored = TraceView.from_dict(d)
        assert restored.name == tv.name
        assert restored.trace_id == tv.trace_id
        assert len(restored.ranges) == 2
        assert restored.ranges[0].label == "Summary"
        assert restored.ranges[1].to_selector.span_type == "TOOL"
        assert restored.ranges[1].input_path == "$.reasoning"
        assert restored.created_by == tv.created_by
        assert restored.view_id == tv.view_id

    def test_to_dict_round_trip_no_ranges(self):
        tv = TraceView(name="empty", trace_id="t")
        d = tv.to_dict()
        restored = TraceView.from_dict(d)
        assert restored.ranges == []
