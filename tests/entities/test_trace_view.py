from __future__ import annotations

import json

import pytest

from mlflow.entities.trace_view import SpanFilter, TraceView
from mlflow.exceptions import MlflowException


class TestSpanFilter:
    def test_creation_defaults(self):
        sf = SpanFilter()
        assert sf.span_name is None
        assert sf.span_type is None
        assert sf.attribute_key is None
        assert sf.attribute_value is None

    def test_creation_with_values(self):
        sf = SpanFilter(
            span_name="my_span",
            span_type="LLM",
            attribute_key="model",
            attribute_value="gpt-4",
        )
        assert sf.span_name == "my_span"
        assert sf.span_type == "LLM"
        assert sf.attribute_key == "model"
        assert sf.attribute_value == "gpt-4"

    def test_to_dict(self):
        sf = SpanFilter(span_name="my_span", span_type="LLM")
        d = sf.to_dict()
        assert d == {
            "span_name": "my_span",
            "span_type": "LLM",
            "attribute_key": None,
            "attribute_value": None,
        }

    def test_from_dict(self):
        d = {"span_name": "x", "span_type": "CHAIN", "attribute_key": "k", "attribute_value": "v"}
        sf = SpanFilter.from_dict(d)
        assert sf.span_name == "x"
        assert sf.span_type == "CHAIN"
        assert sf.attribute_key == "k"
        assert sf.attribute_value == "v"

    def test_from_dict_partial(self):
        sf = SpanFilter.from_dict({"span_name": "only_name"})
        assert sf.span_name == "only_name"
        assert sf.span_type is None

    def test_json_round_trip(self):
        sf = SpanFilter(span_name="test", attribute_key="key")
        json_str = sf.to_json()
        restored = SpanFilter.from_json(json_str)
        assert restored == sf

    def test_to_json_is_valid_json(self):
        sf = SpanFilter(span_name="test")
        parsed = json.loads(sf.to_json())
        assert isinstance(parsed, dict)


class TestTraceView:
    def test_trace_scoped_creation(self):
        tv = TraceView(name="my_view", trace_id="tr-123")
        assert tv.name == "my_view"
        assert tv.trace_id == "tr-123"
        assert tv.experiment_id is None

    def test_experiment_scoped_creation(self):
        tv = TraceView(name="exp_view", experiment_id="exp-456")
        assert tv.name == "exp_view"
        assert tv.experiment_id == "exp-456"
        assert tv.trace_id is None

    def test_scope_property_trace(self):
        tv = TraceView(name="v", trace_id="t1")
        assert tv.scope == "trace"

    def test_scope_property_experiment(self):
        tv = TraceView(name="v", experiment_id="e1")
        assert tv.scope == "experiment"

    def test_validate_scope_both_set_raises(self):
        tv = TraceView(name="v", trace_id="t1", experiment_id="e1")
        with pytest.raises(MlflowException, match="Exactly one of"):
            tv.validate_scope()

    def test_validate_scope_neither_set_raises(self):
        tv = TraceView(name="v")
        with pytest.raises(MlflowException, match="Exactly one of"):
            tv.validate_scope()

    def test_validate_scope_valid_trace(self):
        tv = TraceView(name="v", trace_id="t1")
        tv.validate_scope()

    def test_validate_scope_valid_experiment(self):
        tv = TraceView(name="v", experiment_id="e1")
        tv.validate_scope()

    def test_to_dict_round_trip_trace_scoped(self):
        tv = TraceView(
            name="view1",
            trace_id="tr-1",
            span_filter=SpanFilter(span_name="s1"),
            input_path="input.field",
            output_path="output.field",
            created_by="user@example.com",
            description="A test view",
            view_id="vid-1",
            create_time_ms=1000,
            last_update_time_ms=2000,
        )
        d = tv.to_dict()
        restored = TraceView.from_dict(d)
        assert restored.name == tv.name
        assert restored.trace_id == tv.trace_id
        assert restored.experiment_id is None
        assert restored.span_filter == tv.span_filter
        assert restored.input_path == tv.input_path
        assert restored.output_path == tv.output_path
        assert restored.created_by == tv.created_by
        assert restored.description == tv.description
        assert restored.view_id == tv.view_id
        assert restored.create_time_ms == tv.create_time_ms
        assert restored.last_update_time_ms == tv.last_update_time_ms

    def test_to_dict_round_trip_experiment_scoped(self):
        tv = TraceView(name="view2", experiment_id="exp-1")
        restored = TraceView.from_dict(tv.to_dict())
        assert restored.name == tv.name
        assert restored.experiment_id == tv.experiment_id
        assert restored.trace_id is None

    def test_to_dict_without_span_filter(self):
        tv = TraceView(name="v", trace_id="t")
        d = tv.to_dict()
        assert d["span_filter"] is None

    def test_from_dict_without_span_filter(self):
        d = {"name": "v", "trace_id": "t"}
        tv = TraceView.from_dict(d)
        assert tv.span_filter is None
