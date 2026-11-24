"""Pytest plugin for ``@mlflow.test`` + ``mlflow.genai.evaluate``.

Opt-in: the plugin is intentionally not auto-registered (loading it would make
every pytest run on the machine import mlflow at startup). Enable it by adding
the following to your root ``conftest.py``::

    pytest_plugins = ["mlflow.pytest.plugin"]

or by running pytest with ``-p mlflow.pytest.plugin``. ``@mlflow.test``-marked
tests raise a clear error if the plugin is not enabled.

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

_logger = logging.getLogger(__name__)


def _case_id(item: pytest.Item) -> str | None:
    # callspec.id is the parametrize id pytest itself assigns; parsing it out of
    # item.name with a regex breaks when a param value contains "[" or "]".
    callspec = getattr(item, "callspec", None)
    return callspec.id if callspec is not None else None


def pytest_sessionstart(session: pytest.Session) -> None:
    _session.set_plugin_active()
    _session.reset()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
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
    # across files; the UI renders this nodeid as-is.
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
