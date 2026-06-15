"""``@mlflow.test`` marker.

Marks a test for the (opt-in) MLflow pytest plugin, which sets up the test run
and enables tracing for the marked test. Enable the plugin by adding
``pytest_plugins = ["mlflow.pytest.plugin"]`` to your root ``conftest.py``, or
by running pytest with ``-p mlflow.pytest.plugin``.

    @mlflow.test
    def test_recommendations(agent):
        result = mlflow.genai.evaluate(
            predict_fn=agent.invoke,
            data=[{"inputs": {"prompt": "What tracing tool?"}}],
            scorers=[Guidelines(guidelines="Recommends MLflow"), Safety()],
        )
        assert result.passed, result.reason
"""

from __future__ import annotations

import functools
import os
from typing import Callable, ParamSpec, TypeVar

from mlflow.utils.annotations import experimental

_P = ParamSpec("_P")
_R = TypeVar("_R")

MLFLOW_TEST_ATTR = "_mlflow_test"

_PLUGIN_NOT_ENABLED_MESSAGE = (
    "@mlflow.test requires the MLflow pytest plugin, which is not enabled in "
    "this pytest run. Enable it by adding "
    'pytest_plugins = ["mlflow.pytest.plugin"] to your root conftest.py, or by '
    "running pytest with `-p mlflow.pytest.plugin`."
)


def _ensure_plugin_active() -> None:
    # Only meaningful under pytest; calling the function directly elsewhere
    # (e.g. a notebook) is fine.
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return

    # Imported lazily to keep `import mlflow` (which imports this module) from
    # touching the session module at import time.
    from mlflow.pytest import session

    if not session.is_plugin_active():
        raise RuntimeError(_PLUGIN_NOT_ENABLED_MESSAGE)


@experimental(version="3.14.0")
def test(fn: Callable[_P, _R] | None = None) -> Callable[_P, _R]:
    """Mark a test function for the MLflow pytest plugin.

    Supports both bare ``@mlflow.test`` and ``@mlflow.test()``. Raises at test
    time if the plugin is not enabled, instead of silently running the test
    without MLflow run/trace management.
    """

    def decorator(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(f)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            _ensure_plugin_active()
            return f(*args, **kwargs)

        setattr(wrapper, MLFLOW_TEST_ATTR, True)
        return wrapper

    return decorator if fn is None else decorator(fn)
