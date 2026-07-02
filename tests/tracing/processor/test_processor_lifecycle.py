"""Tests for deferred, drain-gated reclamation of orphaned span processors (Phase 1).

When a tracer provider is swapped (disable()/enable(), set_experiment(), set_destination() in
isolated mode), the outgoing provider's BatchSpanProcessor daemon thread must not be leaked, but
it also must not be shut down while an enclosing trace's spans are still open on it. These tests
exercise both properties.
"""

import threading
import time

import pytest

import mlflow
from mlflow.tracing.processor import base_mlflow
from mlflow.tracing.processor.base_mlflow import (
    _orphaned_processors,
    reclaim_orphaned_processors,
)
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.provider import trace_disabled

from tests.tracing.helper import get_traces


@pytest.fixture(autouse=True)
def enable_batch_processor(monkeypatch):
    # The leak only manifests when the MLflow processor owns a BatchSpanProcessor daemon thread.
    monkeypatch.setenv("MLFLOW_USE_BATCH_SPAN_PROCESSOR", "true")
    yield
    # Reclaim any orphans this test produced so they do not leak into other tests.
    reclaim_orphaned_processors()


def _otel_threads():
    return sum("OtelBatchSpanRecordProcessor" in t.name for t in threading.enumerate())


def _settle(timeout_secs=5.0):
    """Give the reaper time to reclaim all drained orphans."""
    mlflow.flush_trace_async_logging()
    reclaim_orphaned_processors()
    deadline = time.time() + timeout_secs
    while _orphaned_processors and time.time() < deadline:
        time.sleep(0.05)
        reclaim_orphaned_processors()


def test_disable_enable_cycles_do_not_leak_threads():
    # Settle any orphans left over from prior tests so the baseline reflects only threads
    # this test cannot control (the absolute count is shared process-wide across tests).
    mlflow.tracing.enable()
    _settle()
    baseline = _otel_threads()
    for _ in range(10):
        mlflow.tracing.disable()
        mlflow.tracing.enable()
    mlflow.tracing.disable()
    _settle()
    # 10 disable/enable cycles must not grow the live BSP thread count. Without the fix this
    # grows by ~1 per cycle; with it the swapped-out processors are reclaimed.
    assert _otel_threads() <= baseline + 1


def test_set_experiment_swaps_do_not_leak_threads():
    mlflow.tracing.enable()

    @mlflow.trace
    def f():
        return 1

    f()
    _settle()
    baseline = _otel_threads()
    for i in range(10):
        mlflow.set_experiment(f"exp_lifecycle_{i}")
        f()
    _settle()
    # One live provider legitimately keeps one BSP; the swapped-out ones must be reclaimed.
    assert _otel_threads() <= baseline + 1


def test_nested_trace_disabled_inside_active_trace_preserves_outer_trace():
    """The rejected-synchronous-shutdown regression guard.

    A @trace_disabled call nested inside an active @trace swaps the provider mid-trace. The
    drain gate must keep the outgoing provider alive until the enclosing trace ends, so the
    outer trace is not dropped. Also asserts threads stay bounded.
    """
    mlflow.tracing.enable()

    @trace_disabled
    def load_model():
        return 1

    @mlflow.trace
    def agent():
        load_model()
        return "done"

    baseline = _otel_threads()
    n = 5
    for _ in range(n):
        agent()

    mlflow.flush_trace_async_logging()
    traces = get_traces()
    assert len(traces) == n

    _settle()
    assert _otel_threads() <= baseline + 1


def test_drain_gate_blocks_shutdown_while_root_span_open():
    """Directly exercise the drain gate: an orphaned processor with an open in-flight root span
    must NOT be shut down until that span ends.
    """
    exporter = _FakeExporter()
    processor = MlflowV3SpanProcessor(
        span_exporter=exporter,
        export_metrics=False,
        use_batch_processor=True,
    )
    try:
        # Simulate an open root span (enclosing trace in flight) on this processor.
        with processor._deduplication_lock:
            processor._active_root_spans = 1

        # Orphan it (as a provider swap would) and try to reclaim.
        base_mlflow._register_orphaned_processor(processor)
        reclaim_orphaned_processors()

        # Gate is closed: still orphaned, not shut down.
        assert processor in _orphaned_processors
        assert not exporter.shutdown_called

        # Close the root span; now the gate opens.
        with processor._deduplication_lock:
            processor._active_root_spans = 0

        reclaim_orphaned_processors()
        assert processor not in _orphaned_processors
        assert exporter.shutdown_called
    finally:
        with base_mlflow._orphaned_processors_lock:
            if processor in _orphaned_processors:
                _orphaned_processors.remove(processor)
        if processor._batch_delegate is not None:
            processor.shutdown()


def test_drain_gate_blocks_shutdown_while_on_end_in_flight():
    exporter = _FakeExporter()
    processor = MlflowV3SpanProcessor(
        span_exporter=exporter,
        export_metrics=False,
        use_batch_processor=True,
    )
    try:
        with processor._pending_on_end_condition:
            processor._pending_on_end_count = 1

        base_mlflow._register_orphaned_processor(processor)
        reclaim_orphaned_processors()
        assert processor in _orphaned_processors
        assert not exporter.shutdown_called

        with processor._pending_on_end_condition:
            processor._pending_on_end_count = 0

        reclaim_orphaned_processors()
        assert processor not in _orphaned_processors
        assert exporter.shutdown_called
    finally:
        with base_mlflow._orphaned_processors_lock:
            if processor in _orphaned_processors:
                _orphaned_processors.remove(processor)
        if processor._batch_delegate is not None:
            processor.shutdown()


class _FakeExporter:
    def __init__(self):
        self.shutdown_called = False

    def export(self, spans):
        pass

    def shutdown(self):
        self.shutdown_called = True

    def force_flush(self, timeout_millis=30000):
        return True
