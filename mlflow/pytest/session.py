"""Session state for the ``@mlflow.test`` pytest plugin.

Manages the single test run per pytest session and tracks which
test is currently executing (for trace tagging). No ``pytest`` import so it
can be used from notebooks/scripts too.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import uuid

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_TEST

_logger = logging.getLogger(__name__)

TAG_TEST_NAME = "mlflow.test.name"
TAG_SESSION_ID = "mlflow.test.session_id"
TAG_CASE_ID = "mlflow.test.case_id"

_lock = threading.Lock()
_session_id: str | None = None
_run_id: str | None = None
_run_owned: bool = False
_any_test_failed: bool = False
_num_tests: int = 0
_total_test_ms: int = 0

_current = threading.local()


def set_current_test(test_name: str | None, case_id: str | None = None) -> None:
    _current.value = (test_name, case_id)


def current_test() -> tuple[str | None, str | None]:
    return getattr(_current, "value", (None, None))


def reset(session_id: str | None = None) -> None:
    global _session_id, _run_id, _run_owned, _any_test_failed, _num_tests, _total_test_ms
    if session_id is None:
        session_id = os.environ.get("_MLFLOW_TEST_SESSION_ID")
    if not session_id:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        session_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    with _lock:
        _session_id = session_id
        _run_id = None
        _run_owned = False
        _any_test_failed = False
        _num_tests = 0
        _total_test_ms = 0


def session_id() -> str:
    if _session_id is None:
        reset()
    return _session_id


def run_id() -> str | None:
    return _run_id


def record_test(*, failed: bool, duration_ms: int) -> None:
    """Record the outcome and duration of one ``@mlflow.test``-marked test."""
    global _any_test_failed, _num_tests, _total_test_ms
    with _lock:
        _num_tests += 1
        _total_test_ms += duration_ms
        if failed:
            _any_test_failed = True


def num_tests() -> int:
    return _num_tests


def total_test_ms() -> int:
    return _total_test_ms


def ensure_run() -> str | None:
    """Open (or adopt) the test run, once per session. Thread-safe."""
    global _run_id, _run_owned
    with _lock:
        if _run_id is not None:
            return _run_id

        tags = {MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_TEST, TAG_SESSION_ID: session_id()}

        try:
            active = mlflow.active_run()
        except Exception as e:
            _logger.warning("mlflow.test: could not check active run: %s", e)
            return None

        if active is not None:
            try:
                client = MlflowClient()
                for k, v in tags.items():
                    client.set_tag(active.info.run_id, k, v)
            except Exception as e:
                _logger.warning("mlflow.test: could not tag active run: %s", e)
                return None
            _run_id = active.info.run_id
            _run_owned = False
            return _run_id

        try:
            run = mlflow.start_run(tags=tags)
        except Exception as e:
            _logger.warning("mlflow.test: could not start test run: %s", e)
            return None
        _run_id = run.info.run_id
        _run_owned = True
        return _run_id


def finalize() -> None:
    """End the test run, marking it FAILED only if a ``@mlflow.test`` test failed.

    The run status reflects only ``@mlflow.test``-marked tests, not other tests
    that happen to run in the same pytest session.
    """
    global _run_id, _run_owned
    with _lock:
        if _run_id is None:
            return
        run_id_, run_owned, failed = _run_id, _run_owned, _any_test_failed
        _run_id = None
        _run_owned = False

    if run_owned:
        status = "FAILED" if failed else "FINISHED"
        try:
            mlflow.end_run(status=status)
        except Exception as e:
            _logger.warning("Failed to end run %s: %s", run_id_, e)
