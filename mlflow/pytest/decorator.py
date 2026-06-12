"""``@mlflow.test`` marker.

A near no-op decorator whose only job is to be visible at collection time so
the pytest plugin can set up the test run and enable tracing for
the marked test.

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
from typing import Callable, ParamSpec, TypeVar

from mlflow.utils.annotations import experimental

_P = ParamSpec("_P")
_R = TypeVar("_R")

MLFLOW_TEST_ATTR = "_mlflow_test"


@experimental(version="3.14.0")
def test(fn: Callable[_P, _R] | None = None) -> Callable[_P, _R]:
    """Mark a test function for the MLflow pytest plugin.

    Supports both bare ``@mlflow.test`` and ``@mlflow.test()``.
    """

    def decorator(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(f)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return f(*args, **kwargs)

        setattr(wrapper, MLFLOW_TEST_ATTR, True)
        return wrapper

    return decorator if fn is None else decorator(fn)
