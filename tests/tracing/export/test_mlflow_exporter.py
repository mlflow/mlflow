import time
from threading import Thread
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import ReadableSpan

from mlflow.tracing.export.mlflow import InMemoryTraceDataAggregator, MLflowSpanExporter
from mlflow.tracing.types.model import Span, SpanContext, SpanType, Status, StatusCode
from mlflow.tracing.types.wrapper import MLflowSpanWrapper


def test_export():
    trace_id = "trace_id"
    otel_span_root = ReadableSpan(
        name="test_span",
        context=SpanContext(trace_id, "span_id_1"),
        parent=None,
        attributes={
            "key1": "value1",
        },
        start_time=0,
        end_time=4,
    )

    otel_span_child_1 = ReadableSpan(
        name="test_span_child_1",
        context=SpanContext(trace_id, "span_id_2"),
        parent=otel_span_root.context,
        attributes={
            "key2": "value2",
        },
        start_time=1,
        end_time=2,
    )

    otel_span_child_2 = ReadableSpan(
        name="test_span_child_2",
        context=SpanContext(trace_id, "span_id_3"),
        parent=otel_span_root.context,
        start_time=2,
        end_time=3,
    )

    mock_client = MagicMock()
    exporter = MLflowSpanExporter(mock_client)

    # Export the first child span -> no client call
    exporter.export([MLflowSpanWrapper(otel_span_child_1)])

    assert mock_client.log_trace.call_count == 0
    assert len(exporter._trace_aggregator._traces[trace_id].spans) == 1

    # Export the second child span -> no client call
    exporter.export([MLflowSpanWrapper(otel_span_child_2)])

    assert mock_client.log_trace.call_count == 0
    assert len(exporter._trace_aggregator._traces[trace_id].spans) == 2

    # Export the root span -> client call
    root_span = MLflowSpanWrapper(otel_span_root)
    root_span.set_inputs({"input1": "very long input" * 100})
    root_span.set_outputs({"output1": "very long output" * 100})
    exporter.export([root_span])

    assert mock_client.log_trace.call_count == 1
    client_call_args = mock_client.log_trace.call_args[0][0]

    # Trace info should inherit fields from the root span
    trace_info = client_call_args.trace_info
    assert trace_info.trace_id == trace_id
    assert trace_info.name == "test_span"
    assert trace_info.start_time == 0
    assert trace_info.end_time == 4

    # Inputs and outputs in TraceInfo should be serialized and truncated
    assert trace_info.inputs.startswith("{'input1': 'very long input")
    assert len(trace_info.inputs) == 300
    assert trace_info.outputs.startswith("{'output1': 'very long output")
    assert len(trace_info.outputs) == 300

    # All 3 spans should be in the logged trace data
    assert len(client_call_args.trace_data.spans) == 3

    # Spans should be cleared from the aggregator
    assert len(exporter._trace_aggregator._traces) == 0


def test_aggregator_singleton():
    obj1 = InMemoryTraceDataAggregator.get_instance()
    obj2 = InMemoryTraceDataAggregator.get_instance()
    assert obj1 is obj2


def test_aggregator_add_and_pop_span():
    aggregator = InMemoryTraceDataAggregator.get_instance()

    trace_id_1 = "trace_1"
    span_1_1 = _create_test_span(trace_id_1, "span_1_1")
    span_1_1_1 = _create_test_span(trace_id_1, "span_1_1_1", parent_span_id="span_1_1")
    span_1_1_2 = _create_test_span(trace_id_1, "span_1_1_2", parent_span_id="span_1_1")

    # Add a span for a new trace
    aggregator.add_span(span_1_1)

    assert trace_id_1 in aggregator._traces
    assert len(aggregator._traces[trace_id_1].spans) == 1

    # Add more spans to the same trace
    aggregator.add_span(span_1_1_1)
    aggregator.add_span(span_1_1_2)

    assert len(aggregator._traces[trace_id_1].spans) == 3

    # Add a span for another trace
    trace_id_2 = "trace_2"
    span_2_1 = _create_test_span(trace_id_2, "span_2_1")
    span_2_1_1 = _create_test_span(trace_id_2, "span_2_1_1", parent_span_id="span_2_1")

    aggregator.add_span(span_2_1)
    aggregator.add_span(span_2_1_1)

    assert trace_id_2 in aggregator._traces
    assert len(aggregator._traces[trace_id_2].spans) == 2

    # Pop the trace data
    trace_data = aggregator.pop_trace(trace_id_1)
    assert trace_data is not None
    assert len(trace_data.spans) == 3
    assert trace_id_1 not in aggregator._traces

    trace_data = aggregator.pop_trace(trace_id_2)
    assert trace_data is not None
    assert len(trace_data.spans) == 2
    assert trace_id_2 not in aggregator._traces

    # Pop a trace that does not exist
    assert aggregator.pop_trace("trace_3") is None


def test_aggregator_add_and_pop_span_thread_safety():
    aggregator = InMemoryTraceDataAggregator.get_instance()

    # Add spans from 10 different threads to 5 different traces
    trace_ids = [f"trace_{i}" for i in range(5)]
    num_threads = 10

    def add_spans(thread_id):
        for trace_id in trace_ids:
            aggregator.add_span(_create_test_span(trace_id, f"span_{thread_id}"))

    threads = [Thread(target=add_spans, args=[i]) for i in range(num_threads)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for trace_id in trace_ids:
        trace_data = aggregator.pop_trace(trace_id)
        assert trace_data is not None
        assert len(trace_data.spans) == num_threads


def _create_test_span(trace_id, span_id, parent_span_id=None, start_time=None, end_time=None):
    if start_time is None:
        start_time = time.time_ns()
    if end_time is None:
        end_time = time.time_ns()

    return Span(
        name="test_span",
        parent_span_id=parent_span_id,
        context=SpanContext(trace_id, span_id),
        span_type=SpanType.UNKNOWN,
        status=Status(
            status_code=StatusCode.OK,
            description="",
        ),
        start_time=start_time,
        end_time=end_time,
    )
