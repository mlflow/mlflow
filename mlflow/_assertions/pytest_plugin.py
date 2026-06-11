"""Pytest plugin for ``@mlflow.test`` + ``mlflow.genai.evaluate``.

Auto-registered via the ``pytest11`` entry point in ``pyproject.toml``.

What it does:
- Creates one **regression-test run** per pytest session. ``evaluate()``
  inside the test body inherits this active run, so traces and feedback
  attach to it naturally.
- Enables tracing autologging for ``@mlflow.test``-marked tests, the same
  way ``mlflow.genai.evaluate`` does.
- Set ``MLFLOW_TEST_SESSION_ID`` to override the auto-generated session id
  (useful in CI).
"""

from __future__ import annotations

import logging
import re

import pytest

from mlflow._assertions import session as _session
from mlflow._assertions.decorator import MLFLOW_TEST_ATTR

_logger = logging.getLogger(__name__)


def _case_id_from_item_name(item_name: str) -> str | None:
    m = re.search(r"\[(.+)\]$", item_name)
    return m.group(1) if m else None


def pytest_sessionstart(session: pytest.Session) -> None:
    _session.reset()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _session.finalize(exitstatus)


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

    test_name = item.function.__name__
    case_id = _case_id_from_item_name(item.name)

    _session.set_current_test(test_name, case_id)
    _session.ensure_run()

    yield

    _session.set_current_test(None, None)
