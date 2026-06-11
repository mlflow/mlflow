"""Session state for the ``@mlflow.test`` pytest plugin.

Manages the single regression-test run per pytest session and tracks which
test is currently executing (for trace tagging). No ``pytest`` import so it
can be used from notebooks/scripts too.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import uuid

_logger = logging.getLogger(__name__)

TAG_TEST_NAME = "mlflow.test.name"
TAG_SESSION_ID = "mlflow.test.session_id"
TAG_CASE_ID = "mlflow.test.case_id"

_lock = threading.Lock()
_session_id: str | None = None
_run_id: str | None = None
_run_owned: bool = False

_current = threading.local()


def set_current_test(test_name: str | None, case_id: str | None = None) -> None:
    _current.value = (test_name, case_id)


def current_test() -> tuple[str | None, str | None]:
    return getattr(_current, "value", (None, None))


def reset(session_id: str | None = None) -> None:
    global _session_id, _run_id, _run_owned
    if session_id is None:
        session_id = os.environ.get("MLFLOW_TEST_SESSION_ID")
    if not session_id:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        session_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    _session_id = session_id
    _run_id = None
    _run_owned = False


def session_id() -> str:
    if _session_id is None:
        reset()
    return _session_id


def run_id() -> str | None:
    return _run_id


def ensure_run() -> str | None:
    """Open (or adopt) the test run, once per session. Thread-safe."""
    global _run_id, _run_owned
    with _lock:
        if _run_id is not None:
            return _run_id

        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_TEST

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


def finalize(exitstatus: int) -> None:
    global _run_id, _run_owned
    if _run_id is None:
        return

    import mlflow

    if _run_owned:
        status = "FINISHED" if exitstatus == 0 else "FAILED"
        try:
            mlflow.end_run(status=status)
        except Exception as e:
            _logger.warning("Failed to end run %s: %s", _run_id, e)
    _run_id = None
    _run_owned = False
