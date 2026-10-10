import threading
from unittest import mock

import pytest
import requests
import urllib3.exceptions
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

from mlflow.entities.span import Span
from mlflow.entities.trace_location import UCSchemaLocation, UnityCatalog
from mlflow.environment_variables import MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT
from mlflow.tracing.export.databricks_otel_client import (
    DatabricksOtelTokenError,
    DatabricksOtelTokenRefreshError,
    DatabricksOtelUnavailableError,
)
from mlflow.tracing.export.databricks_otel_collector import (
    DatabricksOtelExporter,
    DatabricksOtelSerializationError,
    _is_connection_not_established,
    get_databricks_otel_exporter,
)
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.export.utils import flush_exporter
from mlflow.tracing.fluent import _flush_pending_async_trace_writes
from mlflow.tracing.processor.base_mlflow import BaseMlflowSpanProcessor, retire_batch_processor
from mlflow.tracing.processor.uc_table import DatabricksUCTableSpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v4

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info_with_uc_table

_MODULE = "mlflow.tracing.export.databricks_otel_collector"
_TABLE = "catalog.schema.otel_spans"


def _response(status_code: int, content: bytes = b""):
    response = mock.MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.content = content
    return response


def _partial_response(rejected_spans: int, error_message: str = "rejected"):
    body = ExportTraceServiceResponse()
    body.partial_success.rejected_spans = rejected_spans
    body.partial_success.error_message = error_message
    return _response(200, body.SerializeToString())


class _Queue:
    def __init__(self):
        self.flush_calls = []

    def flush(self, terminate=False):
        self.flush_calls.append(terminate)


class _Batcher:
    instances = []

    def __init__(self, async_task_queue, log_spans_func):
        self.queue = async_task_queue
        self.log_spans_func = log_spans_func
        self.pending = []
        self.flush_calls = []
        self._stopped = False
        self.__class__.instances.append(self)

    def add_span(self, location, span):
        self.pending.append((location, span))

    def flush(self):
        self.flush_calls.append(False)
        pending = self.pending
        self.pending = []
        if pending:
            self.log_spans_func(pending[0][0], [span for _, span in pending])

    def shutdown(self):
        self.flush_calls.append(True)
        self.flush()
        self._stopped = True


class _MetadataExporter:
    def __init__(self, async_enabled=False):
        self.async_queue = _Queue() if async_enabled else None
        self.async_enabled = async_enabled
        self.export_calls = []
        self.write_calls = []
        self.shutdown_calls = 0

    def should_log_async(self):
        return self.async_enabled

    def export(self, spans):
        self.export_calls.append(list(spans))

    def write_spans_to_table(self, location, spans):
        self.write_calls.append((location, list(spans)))

    def shutdown(self):
        self.shutdown_calls += 1


class _Collector:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.send_calls = []
        self.close_calls = 0

    def export_spans(self, spans):
        self.send_calls.append(list(spans))
        if not self.outcomes:
            return _response(200)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.close_calls += 1


def _make_router(monkeypatch, outcomes=(), async_enabled=False):
    metadata = _MetadataExporter(async_enabled=async_enabled)
    collector = _Collector(outcomes)
    if async_enabled:
        _Batcher.instances.clear()
        monkeypatch.setattr(f"{_MODULE}.SpanBatcher", _Batcher)
    router = DatabricksOtelExporter(
        table_name=_TABLE,
        otel_client=collector,
        fallback_exporter=metadata,
    )
    return router, collector, metadata


def _new_connection_error():
    return requests.ConnectionError(
        urllib3.exceptions.NewConnectionError(None, "Failed to establish a new connection")
    )


def _name_resolution_connection_error():
    return requests.ConnectionError(
        urllib3.exceptions.NameResolutionError(
            "collector.example", None, "Name or service not known"
        )
    )


def _max_retry_new_connection_error():
    return requests.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            None,
            "/v1/traces",
            urllib3.exceptions.NewConnectionError(None, "Failed to establish a new connection"),
        )
    )


def _max_retry_connection_reset_error():
    return requests.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            None, "/v1/traces", urllib3.exceptions.ProtocolError("Connection reset by peer")
        )
    )


def _context_chained_new_connection_error():
    error = requests.ConnectionError("wrapped connection failure")
    error.__context__ = urllib3.exceptions.NewConnectionError(
        None, "Failed to establish a new connection"
    )
    return error


def _connection_reset_error():
    return requests.ConnectionError(urllib3.exceptions.ProtocolError("Connection reset by peer"))


@pytest.mark.parametrize(
    ("error_factory", "expected"),
    [
        (lambda: requests.ConnectTimeout("connect timed out"), True),
        (_new_connection_error, True),
        (_name_resolution_connection_error, True),
        (_max_retry_new_connection_error, True),
        (_max_retry_connection_reset_error, False),
        (_context_chained_new_connection_error, True),
        (_connection_reset_error, False),
        (lambda: requests.ConnectionError("connection refused"), False),
        (lambda: requests.ReadTimeout("read timed out"), False),
        (lambda: requests.Timeout("timed out"), False),
        (lambda: RuntimeError("boom"), False),
    ],
)
def test_is_connection_not_established(error_factory, expected):
    assert _is_connection_not_established(error_factory()) is expected


def test_export_calls_metadata_once_and_routes_public_export(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch)
    spans = [create_mock_otel_span(trace_id=1, span_id=1)]

    router.export(spans)

    assert metadata.export_calls == [spans]
    assert len(collector.send_calls) == 1
    assert metadata.write_calls == []


def test_router_keeps_batcher_when_request_time_async_policy_changes(monkeypatch):
    metadata = _MetadataExporter(async_enabled=False)
    metadata.async_queue = _Queue()
    collector = _Collector()
    _Batcher.instances.clear()
    monkeypatch.setattr(f"{_MODULE}.SpanBatcher", _Batcher)
    router = DatabricksOtelExporter(
        table_name=_TABLE,
        otel_client=collector,
        fallback_exporter=metadata,
    )

    metadata.async_enabled = True
    router.export([create_mock_otel_span(trace_id=70, span_id=1)])

    assert collector.send_calls == []
    flush_exporter(router)
    assert len(collector.send_calls) == 1


def test_export_routes_before_metadata_and_exports_metadata_for_empty_batch(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch)
    events = []
    collector.export_spans = lambda spans: events.append("collector") or _response(200)
    metadata.export = lambda spans: events.append("metadata")

    router.export([create_mock_otel_span(trace_id=16, span_id=1)])
    router.export([])

    assert events == ["collector", "metadata", "metadata"]
    assert len(collector.send_calls) == 0


@pytest.mark.parametrize("status_code", [503, 504])
def test_async_server_failure_replays_after_flush_and_pins_rest(monkeypatch, status_code):
    router, collector, metadata = _make_router(
        monkeypatch,
        outcomes=[_response(status_code, b"gateway failure"), _response(200)],
        async_enabled=True,
    )
    first = [create_mock_otel_span(trace_id=2, span_id=2)]
    second = [create_mock_otel_span(trace_id=3, span_id=3)]

    router.export(first)
    assert collector.send_calls == []
    router.flush()
    assert len(collector.send_calls) == 1
    assert len(metadata.write_calls) == 1
    assert router.using_legacy_exporter

    router.export(second)
    router.flush()
    assert len(collector.send_calls) == 1
    assert len(metadata.write_calls) == 2
    assert len(metadata.export_calls) == 2


@pytest.mark.parametrize(
    ("outcome", "sticky"),
    [
        (_response(408), False),
        (_response(413), False),
        (_response(429), False),
        (_response(400, b"bad request"), True),
        (_response(403, b"forbidden"), True),
        (_response(404, b"not found"), True),
        (_response(401, b"unauthorized"), True),
        (DatabricksOtelTokenError("mint failed"), False),
        (DatabricksOtelTokenRefreshError("refresh failed"), True),
        (_connection_reset_error(), True),
        (requests.ReadTimeout("read timed out"), True),
        (_response(500, b"server error"), True),
        (_response(502, b"bad gateway"), True),
        (_response(503, b"service unavailable"), True),
        (_response(504, b"gateway timeout"), True),
    ],
)
def test_routing_matrix(monkeypatch, outcome, sticky):
    router, collector, metadata = _make_router(monkeypatch, outcomes=[outcome, _response(200)])
    first = [create_mock_otel_span(trace_id=4, span_id=4)]
    second = [create_mock_otel_span(trace_id=5, span_id=5)]

    with mock.patch(f"{_MODULE}._logger") as logger:
        router.export(first)
        router.export(second)

    assert len(metadata.write_calls) == (2 if sticky else 1)
    assert len(collector.send_calls) == (1 if sticky else 2)
    assert router.using_legacy_exporter is sticky
    assert metadata.write_calls[0][1][0].span_id == Span(first[0]).span_id

    if getattr(outcome, "status_code", None) in (500, 502, 503, 504):
        warning = " ".join(str(arg) for arg in logger.warning.call_args.args)
        assert "duplicate" in warning.lower()


def test_connection_establishment_failure_is_sticky_safe_replay(monkeypatch):
    router, collector, metadata = _make_router(
        monkeypatch,
        outcomes=[requests.ConnectTimeout("connect timed out"), _response(200)],
    )
    router.export([create_mock_otel_span(trace_id=6, span_id=6)])
    router.export([create_mock_otel_span(trace_id=7, span_id=7)])

    assert len(collector.send_calls) == 1
    assert len(metadata.write_calls) == 2
    assert router.using_legacy_exporter


def test_concurrent_ambiguous_failures_replay_both_batches_and_warn_once(monkeypatch):
    router, collector, metadata = _make_router(
        monkeypatch,
        outcomes=[_response(503, b"server error"), _connection_reset_error()],
    )
    collector.barrier = threading.Barrier(2)

    def concurrent_export_spans(spans):
        collector.send_calls.append(list(spans))
        collector.barrier.wait(timeout=5)
        outcome = collector.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    collector.export_spans = concurrent_export_spans
    spans = [
        [create_mock_otel_span(trace_id=71, span_id=1)],
        [create_mock_otel_span(trace_id=72, span_id=2)],
    ]

    with mock.patch(f"{_MODULE}._logger") as logger:
        threads = [
            threading.Thread(target=router.export, args=(span,), name=f"router-export-{index}")
            for index, span in enumerate(spans)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        router.export([create_mock_otel_span(trace_id=73, span_id=3)])

    assert all(not thread.is_alive() for thread in threads)
    assert len(collector.send_calls) == 2
    assert len(metadata.write_calls) == 3
    assert router.using_legacy_exporter
    assert logger.warning.call_count == 1
    warning = logger.warning.call_args.args[0]
    assert "duplicate" in warning
    assert "concurrent exports already in progress may still contact the collector" in warning


@pytest.mark.parametrize("rejected_spans", [2])
def test_successful_200_all_rejected_replays_and_pins_rest(monkeypatch, rejected_spans):
    router, collector, metadata = _make_router(
        monkeypatch,
        outcomes=[_partial_response(rejected_spans), _response(200)],
    )
    router.export([
        create_mock_otel_span(trace_id=8, span_id=1),
        create_mock_otel_span(trace_id=8, span_id=2),
    ])
    router.export([create_mock_otel_span(trace_id=9, span_id=3)])

    assert len(collector.send_calls) == 1
    assert len(metadata.write_calls) == 2
    assert router.using_legacy_exporter


@pytest.mark.parametrize("rejected_spans", [1, -1, 3])
def test_successful_200_partial_or_invalid_drops_current_and_pins_rest(monkeypatch, rejected_spans):
    router, collector, metadata = _make_router(
        monkeypatch,
        outcomes=[_partial_response(rejected_spans), _response(200)],
    )
    router.export([
        create_mock_otel_span(trace_id=10, span_id=1),
        create_mock_otel_span(trace_id=10, span_id=2),
    ])
    router.export([create_mock_otel_span(trace_id=11, span_id=3)])

    assert len(collector.send_calls) == 1
    assert len(metadata.write_calls) == 1
    assert metadata.write_calls[0][0] == _TABLE
    assert router.using_legacy_exporter


def test_successful_200_invalid_body_drops_current_and_pins_rest(monkeypatch):
    router, collector, metadata = _make_router(
        monkeypatch,
        outcomes=[_response(200, b"malformed response"), _response(200)],
    )
    with mock.patch(f"{_MODULE}._logger") as logger:
        router.export([create_mock_otel_span(trace_id=12, span_id=1)])
        router.export([create_mock_otel_span(trace_id=13, span_id=2)])

    assert len(collector.send_calls) == 1
    assert len(metadata.write_calls) == 1
    assert router.using_legacy_exporter
    warning = " ".join(str(arg) for arg in logger.warning.call_args.args)
    assert "dropped instead of replayed" in warning
    assert "could duplicate" in warning


def test_rest_fallback_always_uses_pinned_table(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch, outcomes=[_response(400)])
    router.export([create_mock_otel_span(trace_id=14, span_id=1)])

    assert collector.send_calls
    assert metadata.write_calls[0][0] == _TABLE


def test_successful_non_200_response_is_accepted(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch, outcomes=[_response(201)])
    router.export([create_mock_otel_span(trace_id=17, span_id=1)])

    assert len(collector.send_calls) == 1
    assert metadata.write_calls == []
    assert not router.using_legacy_exporter


@pytest.mark.parametrize("enabled", [False, True])
def test_config_unavailable_warning_follows_explicit_flag(monkeypatch, enabled):
    router, collector, metadata = _make_router(
        monkeypatch, outcomes=[DatabricksOtelUnavailableError("collector unavailable")]
    )
    if enabled:
        monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "true")
    else:
        monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)

    with mock.patch(f"{_MODULE}._logger") as logger:
        router.export([create_mock_otel_span(trace_id=18, span_id=1)])

    assert len(metadata.write_calls) == 1
    (logger.warning if enabled else logger.debug).assert_called_once()
    (logger.debug if enabled else logger.warning).assert_not_called()


def test_serialization_failure_does_not_replay_or_pin(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch)
    collector.outcomes = [DatabricksOtelSerializationError("cannot encode")]

    with mock.patch(f"{_MODULE}._logger") as logger:
        router.export([create_mock_otel_span(trace_id=19, span_id=1)])

    assert metadata.write_calls == []
    assert not router.using_legacy_exporter
    logger.warning.assert_called_once()


def test_flush_and_shutdown_drain_components_in_order(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch, async_enabled=True)
    router.export([create_mock_otel_span(trace_id=15, span_id=1)])

    router.flush()
    batcher = _Batcher.instances[0]
    assert batcher.flush_calls == [False]
    assert metadata.async_queue.flush_calls == [False]
    assert len(collector.send_calls) == 1

    router.shutdown()
    assert metadata.shutdown_calls == 1
    assert collector.close_calls == 1
    assert batcher._stopped
    assert metadata.async_queue.flush_calls[-1] is True


def test_no_batch_processor_flush_drains_router_batcher(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch, async_enabled=True)
    router.export([create_mock_otel_span(trace_id=74, span_id=1)])

    with (
        mock.patch("mlflow.tracing.fluent._get_trace_exporter", return_value=router),
        mock.patch("mlflow.tracing.processor.base_mlflow.flush_all_batch_processors"),
    ):
        _flush_pending_async_trace_writes()

    assert len(collector.send_calls) == 1
    assert metadata.async_queue.flush_calls == [False]


def test_retiring_direct_uc_processor_drains_router_batcher(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch, async_enabled=True)
    processor = DatabricksUCTableSpanProcessor(span_exporter=router)
    router.export([create_mock_otel_span(trace_id=76, span_id=1)])

    retire_batch_processor(processor)

    assert len(collector.send_calls) == 1
    assert metadata.async_queue.flush_calls == [True]
    assert collector.close_calls == 1


def test_shutdown_drains_router_batcher_when_metadata_shutdown_fails(monkeypatch):
    router, collector, metadata = _make_router(monkeypatch, async_enabled=True)
    router.export([create_mock_otel_span(trace_id=75, span_id=1)])
    metadata.shutdown = mock.Mock(side_effect=RuntimeError("metadata shutdown failed"))

    with pytest.raises(RuntimeError, match="metadata shutdown failed"):
        router.shutdown()

    assert len(collector.send_calls) == 1
    assert metadata.async_queue.flush_calls == [True]
    assert collector.close_calls == 1


def _make_real_metadata_router(monkeypatch, outcome):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "true")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "10")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "60000")
    metadata = DatabricksUCTableSpanExporter(tracking_uri="databricks", metadata_only=True)
    metadata._client = mock.MagicMock()
    collector = _Collector([outcome])
    router = DatabricksOtelExporter(
        table_name=_TABLE,
        otel_client=collector,
        fallback_exporter=metadata,
    )
    return router, collector, metadata


def _register_metadata_trace(otel_span):
    trace_manager = InMemoryTraceManager.get_instance()
    trace_id = generate_trace_id_v4(otel_span, "catalog.schema")
    trace_info = create_test_trace_info_with_uc_table(trace_id, "catalog", "schema")
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(Span(otel_span))


@pytest.mark.parametrize("collector_outcome", [_response(200), _response(503, b"unavailable")])
def test_real_metadata_exporter_is_once_and_router_flush_paths_drain(
    monkeypatch, collector_outcome
):
    router, collector, metadata = _make_real_metadata_router(monkeypatch, collector_outcome)
    trace_id = 20 if collector_outcome.status_code == 200 else 21
    otel_span = create_mock_otel_span(trace_id=trace_id, span_id=1)
    _register_metadata_trace(otel_span)
    processor = BaseMlflowSpanProcessor(router, export_metrics=False, use_batch_processor=True)

    try:
        with mock.patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"):
            router.export([otel_span])

        if collector_outcome.status_code == 200:
            flush_exporter(router)
        else:
            retire_batch_processor(processor)

        metadata._client.start_trace.assert_called_once()
        metadata._client._upload_trace_data.assert_not_called()
        expected_rest_calls = 0 if collector_outcome.status_code == 200 else 1
        assert metadata._client.log_spans.call_count == expected_rest_calls
        assert collector.send_calls == [mock.ANY]
    finally:
        router.shutdown()
        if processor._batch_delegate is not None:
            processor.shutdown()


def test_factory_is_default_on_only_for_unity_catalog(monkeypatch):
    destination = UnityCatalog(catalog_name="catalog", schema_name="schema", table_prefix="trace")
    destination._otel_spans_table_name = _TABLE
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)

    with (
        mock.patch(f"{_MODULE}.DatabricksOtelExporter") as exporter_cls,
        mock.patch(f"{_MODULE}.DatabricksOTelClient") as client_cls,
        mock.patch(f"{_MODULE}.DatabricksUCTableSpanExporter") as fallback_cls,
    ):
        result = get_databricks_otel_exporter(destination, "databricks")

    assert result is exporter_cls.return_value
    client_cls.assert_called_once_with(
        tracking_uri="databricks", token_source=None, table_name=_TABLE
    )
    fallback_cls.assert_called_once_with(tracking_uri="databricks", metadata_only=True)
    exporter_cls.assert_called_once_with(
        table_name=_TABLE,
        otel_client=client_cls.return_value,
        fallback_exporter=fallback_cls.return_value,
    )

    schema_destination = UCSchemaLocation(catalog_name="catalog", schema_name="schema")
    assert get_databricks_otel_exporter(schema_destination, "databricks") is None


@pytest.mark.parametrize("enabled", [False, True])
def test_factory_missing_table_warning_follows_explicit_flag(monkeypatch, enabled):
    destination = UnityCatalog(catalog_name="catalog", schema_name="schema", table_prefix="trace")
    if enabled:
        monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "true")
    else:
        monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)

    with mock.patch(f"{_MODULE}._logger") as logger:
        assert get_databricks_otel_exporter(destination, "databricks") is None

    (logger.warning if enabled else logger.debug).assert_called_once()
    (logger.debug if enabled else logger.warning).assert_not_called()


def test_factory_respects_explicit_disable(monkeypatch):
    destination = UnityCatalog(catalog_name="catalog", schema_name="schema", table_prefix="trace")
    destination._otel_spans_table_name = _TABLE
    monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "false")

    with mock.patch(f"{_MODULE}.DatabricksOtelExporter") as exporter_cls:
        assert get_databricks_otel_exporter(destination, "databricks") is None
    exporter_cls.assert_not_called()
