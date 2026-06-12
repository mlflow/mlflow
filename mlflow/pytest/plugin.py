"""Pytest plugin for ``@mlflow.test`` + ``mlflow.genai.evaluate``.

Auto-registered via the ``pytest11`` entry point in ``pyproject.toml``.

What it does:
- Creates one **test run** per pytest session. ``evaluate()``
  inside the test body inherits this active run, so traces and feedback
  attach to it naturally.
- Enables tracing autologging for ``@mlflow.test``-marked tests, the same
  way ``mlflow.genai.evaluate`` does.
"""

from __future__ import annotations

import logging
import time

import pytest
from _pytest.outcomes import Skipped, XFailed

from mlflow.pytest import session as _session
from mlflow.pytest.decorator import MLFLOW_TEST_ATTR
from mlflow.telemetry.events import MlflowTestEvent
from mlflow.telemetry.track import _record_event

_logger = logging.getLogger(__name__)


def _case_id(item: pytest.Item) -> str | None:
    # callspec.id is the parametrize id pytest itself assigns; parsing it out of
    # item.name with a regex breaks when a param value contains "[" or "]".
    callspec = getattr(item, "callspec", None)
    return callspec.id if callspec is not None else None


def pytest_sessionstart(session: pytest.Session) -> None:
    _session.reset()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # One event per pytest session that ran at least one @mlflow.test, capturing
    # how many marked tests ran and their total execution time. Gate on the test
    # count, not run_id: marked tests can run even when run creation failed
    # (e.g. tracking unavailable).
    if _session.num_tests() > 0:
        _record_event(
            MlflowTestEvent,
            {"num_tests": _session.num_tests()},
            duration_ms=_session.total_duration_ms(),
        )
    _session.finalize()


def _is_mlflow_test(item: pytest.Item) -> bool:
    if not isinstance(item, pytest.Function):
        return False
    return getattr(item.function, MLFLOW_TEST_ATTR, False) is True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Set up the MLflow environment for ``@mlflow.test``-marked tests."""
    if not _is_mlflow_test(item):
        yield
        return

    # Full test id (e.g. "tests/foo/test_a.py::test_x[case]") so names are unique
    # across files; the UI renders just the function name.
    _session.set_current_test(item.nodeid, _case_id(item))
    _session.ensure_run()

    start = time.time()
    outcome = yield
    duration_ms = int((time.time() - start) * 1000)

    # Skips/xfails raise their own outcome exceptions; don't count them toward
    # the run outcome or telemetry. Any other exception is a genuine failure.
    exc = outcome.excinfo[1] if outcome.excinfo is not None else None
    if not isinstance(exc, (Skipped, XFailed)):
        _session.record_test(failed=exc is not None, duration_ms=duration_ms)

    _session.set_current_test(None, None)
