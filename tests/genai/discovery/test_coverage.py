from unittest import mock

import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.discovery.coverage import (
    GAP_CACHE_USAGE_MISSING,
    GAP_CONTENT_ELIDED,
    GAP_MODEL_COMPLETION_MISSING,
    GAP_MODEL_PROMPT_MISSING,
    GAP_RETRIEVAL_COUNT_ONLY,
    GAP_ROOT_OPAQUE,
    GAP_TOOL_COUNT_ONLY,
    assess_sample_data_adequacy,
    assess_trace_data_adequacy,
)


def _trace(build, name="agent"):
    @mlflow.trace(name=name, span_type=SpanType.CHAIN)
    def _run():
        return build()

    _run()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def _sound_trace():
    def build():
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": "refund window"})
            span.set_outputs({"documents": ["Refunds are accepted within 30 days."]})
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_inputs({"prompt": "What is the refund window?"})
            span.set_outputs({"text": "You have 30 days."})
        return "You have 30 days."

    return _trace(build)


def test_a_trace_carrying_its_content_has_no_gaps():
    adequacy = assess_trace_data_adequacy(_sound_trace())

    assert adequacy.gaps == []
    assert adequacy.is_thin is False
    assert adequacy.quality_judgeable is True


def test_retrieval_output_that_is_only_a_count_is_not_judgeable():
    def build():
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": "refund window"})
            # The count is not the content: nothing here supports a relevance judgment.
            span.set_outputs("20 matches")
        return "You have 30 days."

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert GAP_RETRIEVAL_COUNT_ONLY in adequacy.gaps
    assert adequacy.quality_judgeable is False
    assert "not the text" in adequacy.describe()


def test_tool_output_that_is_only_a_count_is_reported_as_a_tool_gap():
    def build():
        with mlflow.start_span(name="lookup", span_type=SpanType.TOOL) as span:
            span.set_inputs({"order": "ord-1"})
            span.set_outputs("retrieved 8 rows")
        return "done"

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert GAP_TOOL_COUNT_ONLY in adequacy.gaps
    assert GAP_RETRIEVAL_COUNT_ONLY not in adequacy.gaps


def test_a_real_payload_that_merely_mentions_a_number_is_still_judgeable():
    def build():
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_outputs({"documents": ["Found 20 matches for refunds in the policy index."]})
        return "ok"

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert adequacy.gaps == []


def test_model_span_missing_prompt_or_completion_is_named_separately():
    def build():
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_outputs({"text": "You have 30 days."})
        with mlflow.start_span(name="rewrite", span_type=SpanType.LLM) as span:
            span.set_inputs({"prompt": "shorten this"})
        return "You have 30 days."

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert GAP_MODEL_PROMPT_MISSING in adequacy.gaps
    assert GAP_MODEL_COMPLETION_MISSING in adequacy.gaps
    assert adequacy.quality_judgeable is False


def test_elided_span_content_is_reported():
    def build():
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_inputs({"prompt": "What is the refund window?"})
            span.set_outputs({"text": "[truncated]"})
        return "ok"

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert GAP_CONTENT_ELIDED in adequacy.gaps
    assert adequacy.quality_judgeable is False


def test_opaque_root_payload_is_reported():
    def build():
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_inputs({"prompt": "hello"})
            span.set_outputs({"text": "hi"})
        return "<AgentState object at 0x7f9c1a2b3c40>"

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert GAP_ROOT_OPAQUE in adequacy.gaps
    assert adequacy.quality_judgeable is False


def test_one_reason_is_reported_once_however_many_spans_show_it():
    def build():
        for _ in range(3):
            with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
                span.set_outputs("14 docs")
        return "ok"

    adequacy = assess_trace_data_adequacy(_trace(build))

    assert adequacy.gaps.count(GAP_RETRIEVAL_COUNT_ONLY) == 1


def test_tokens_without_cache_or_cost_blocks_pricing_but_not_quality():
    trace = _sound_trace()
    trace.info.trace_metadata["mlflow.trace.tokenUsage"] = (
        '{"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}'
    )

    adequacy = assess_trace_data_adequacy(trace)

    assert GAP_CACHE_USAGE_MISSING in adequacy.gaps
    assert adequacy.cost_judgeable is False
    # The payload is intact, so correctness and relevance can still be judged.
    assert adequacy.quality_judgeable is True


def test_a_trace_reporting_cost_is_priceable():
    trace = _sound_trace()
    trace.info.trace_metadata["mlflow.trace.tokenUsage"] = (
        '{"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}'
    )
    trace.info.trace_metadata["mlflow.trace.cost"] = '{"total_cost": 0.0042}'

    adequacy = assess_trace_data_adequacy(trace)

    assert GAP_CACHE_USAGE_MISSING not in adequacy.gaps
    assert adequacy.cost_judgeable is True


def test_a_sample_reports_every_trace_so_the_denominator_is_not_shrunk():
    traces = [_sound_trace() for _ in range(3)]

    adequacy = assess_sample_data_adequacy(traces)

    assert len(adequacy) == 3
    assert set(adequacy) == {trace.info.trace_id for trace in traces}


def test_a_trace_whose_shape_cannot_be_read_is_counted_not_dropped():
    # A caller can hand the pipeline trace-like objects (telemetry tests do), and losing one
    # silently would shrink the denominator a failure rate is quoted against.
    traces = [mock.MagicMock(spec="Trace") for _ in range(3)]

    adequacy = assess_sample_data_adequacy(traces)

    assert len(adequacy) == 3
    assert all(not entry.quality_judgeable for entry in adequacy.values())
    assert all("unreadable trace at position" in key for key in adequacy)
