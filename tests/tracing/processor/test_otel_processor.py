from unittest import mock

from opentelemetry.sdk.trace.export import SpanExportResult

from mlflow.entities.trace_info import TraceInfo, TraceLocation, TraceState
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.trace_manager import ManagerTrace

from tests.tracing.helper import create_mock_otel_span


def _make_processor():
    exporter = mock.MagicMock()
    exporter.export.return_value = SpanExportResult.SUCCESS
    processor = OtelSpanProcessor(span_exporter=exporter, export_metrics=False)
    return processor, exporter


def _make_manager_trace(tags: dict[str, str]) -> ManagerTrace:
    from mlflow.entities import Trace, TraceData

    info = TraceInfo(
        trace_id="tr-abc123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        execution_duration=None,
        state=TraceState.OK,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: "3"},
        tags=tags,
    )
    return ManagerTrace(trace=Trace(info=info, data=TraceData()), prompts=[], is_remote_trace=False)


def test_on_end_emits_per_tag_attributes():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    assert root_span.parent is None

    manager_trace = _make_manager_trace(tags={"env": "production", "version": "1.0"})

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=manager_trace):
        processor.on_end(root_span)

    assert root_span._attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "env") == "production"
    assert root_span._attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "version") == "1.0"


def test_on_end_skips_tag_emission_when_no_tags():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    manager_trace = _make_manager_trace(tags={})

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=manager_trace):
        processor.on_end(root_span)

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in root_span._attributes)


def test_on_end_skips_tag_emission_for_child_span():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    child_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x5678, parent_id=0x1234)
    assert child_span.parent is not None

    with mock.patch.object(processor._trace_manager, "pop_trace") as mock_pop:
        processor.on_end(child_span)
        mock_pop.assert_not_called()

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in child_span._attributes)


def test_on_end_handles_missing_trace_gracefully():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=None):
        processor.on_end(root_span)

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in root_span._attributes)


def test_on_end_filters_mlflow_prefixed_tags():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    tags = {
        "user_tag": "keep_me",
        "mlflow.traceName": "my-trace",
        "mlflow.trace.spansLocation": "TRACKING_STORE",
        "mlflow.user": "alice",
        "mlflow.artifactLocation": "s3://bucket",
    }
    manager_trace = _make_manager_trace(tags=tags)

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=manager_trace):
        processor.on_end(root_span)

    assert root_span._attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "user_tag") == "keep_me"
    mlflow_tag_attrs = [
        k
        for k in root_span._attributes
        if k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX)
        and k != SpanAttributeKey.TRACE_TAG_PREFIX + "user_tag"
    ]
    assert mlflow_tag_attrs == []


def test_dual_export_preserves_evaluator_metrics_without_canonical_usage(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "true")
    processor, exporter = _make_processor()
    assert not processor._should_register_traces

    span = create_mock_otel_span(trace_id=0xDEADBEEF, span_id=0x5678, parent_id=0x1234)
    span._attributes = {
        SpanAttributeKey.CHAT_USAGE: '{"input_tokens": 10, "output_tokens": 5}',
        SpanAttributeKey.LLM_COST: '{"total_cost": 0.003}',
        "gen_ai.usage.input_tokens": 10,
    }
    live_span = mock.MagicMock()
    live_span.get_attribute.side_effect = lambda key: {
        SpanAttributeKey.EVALUATION_SCORER: True,
        SpanAttributeKey.CHAT_USAGE: {"input_tokens": 10, "output_tokens": 5},
    }.get(key)
    with (
        mock.patch.object(
            processor._trace_manager, "get_mlflow_trace_id_from_otel_id", return_value="tr-abc"
        ),
        mock.patch.object(processor._trace_manager, "get_span_from_id", return_value=live_span),
    ):
        processor.on_end(span)
        assert processor.force_flush()

    exported_span = exporter.export.call_args.args[0][0]
    assert SpanAttributeKey.CHAT_USAGE not in exported_span.attributes
    assert SpanAttributeKey.LLM_COST not in exported_span.attributes
    assert exported_span.attributes[SpanAttributeKey.EVALUATION_TOKEN_USAGE] == (
        '{"input_tokens": 10, "output_tokens": 5}'
    )
    assert exported_span.attributes[SpanAttributeKey.EVALUATION_COST] == '{"total_cost": 0.003}'
    assert "gen_ai.usage.input_tokens" not in exported_span.attributes
    assert live_span._span._attributes == exported_span.attributes
    live_span.get_attribute.assert_any_call(SpanAttributeKey.EVALUATION_SCORER)
    processor.shutdown()
