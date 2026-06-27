import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import mlflow.spark.autologging as autologging_module
from mlflow.spark.autologging import _stop_listen_for_spark_activity


@pytest.fixture
def autolog_caplog(caplog):
    # The "mlflow" logger sets propagate=False, so caplog's root handler does not see
    # records. Attach caplog's handler directly to the autologging logger.
    logger = logging.getLogger("mlflow.spark.autologging")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="mlflow.spark.autologging"):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _make_spark_context(shutdown_side_effect=None):
    gw = MagicMock()
    if shutdown_side_effect is not None:
        gw.shutdown_callback_server.side_effect = shutdown_side_effect
    sc = MagicMock()
    sc._gateway = gw
    return sc


def test_fast_shutdown_returns_without_warning(autolog_caplog):
    sc = _make_spark_context()
    _stop_listen_for_spark_activity(sc)
    sc._gateway.shutdown_callback_server.assert_called_once()
    assert not any("did not complete" in r.message for r in autolog_caplog.records)


def test_hanging_shutdown_times_out_and_logs_warning(autolog_caplog):
    hang = threading.Event()

    def _hang():
        hang.wait()

    sc = _make_spark_context(shutdown_side_effect=_hang)
    try:
        with patch.object(autologging_module, "_CALLBACK_SERVER_SHUTDOWN_TIMEOUT_SECONDS", 0.2):
            _stop_listen_for_spark_activity(sc)
        warnings = [r for r in autolog_caplog.records if "did not complete" in r.message]
        assert len(warnings) == 1
        assert "CLOSE_WAIT" in warnings[0].message
    finally:
        hang.set()


def test_hanging_shutdown_does_not_block_caller():
    hang = threading.Event()

    def _hang():
        hang.wait()

    sc = _make_spark_context(shutdown_side_effect=_hang)
    try:
        with patch.object(autologging_module, "_CALLBACK_SERVER_SHUTDOWN_TIMEOUT_SECONDS", 0.2):
            start = time.monotonic()
            _stop_listen_for_spark_activity(sc)
            elapsed = time.monotonic() - start
        assert elapsed < 2.0
    finally:
        hang.set()


def test_exception_during_shutdown_logs_error_not_timeout(autolog_caplog):
    sc = _make_spark_context(shutdown_side_effect=Exception("connection reset"))
    with patch.object(autologging_module, "_CALLBACK_SERVER_SHUTDOWN_TIMEOUT_SECONDS", 0.2):
        _stop_listen_for_spark_activity(sc)
    error_warnings = [r for r in autolog_caplog.records if "Failed to shut down" in r.message]
    assert len(error_warnings) == 1
    assert "connection reset" in error_warnings[0].message
    assert not any("did not complete" in r.message for r in autolog_caplog.records)


def test_shutdown_thread_is_daemon():
    created_threads = []
    original_init = threading.Thread.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created_threads.append(self)

    hang = threading.Event()
    sc = _make_spark_context(shutdown_side_effect=lambda: hang.wait())
    try:
        with (
            patch.object(threading.Thread, "__init__", capturing_init),
            patch.object(autologging_module, "_CALLBACK_SERVER_SHUTDOWN_TIMEOUT_SECONDS", 0.1),
        ):
            _stop_listen_for_spark_activity(sc)
        assert created_threads
        assert all(t.daemon for t in created_threads)
    finally:
        hang.set()
