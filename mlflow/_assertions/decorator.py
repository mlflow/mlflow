"""``@mlflow.test`` marker.

A near no-op decorator whose only job is to be visible at collection time so
the pytest plugin can set up the regression-test run and enable tracing for
the marked test. The test body calls ``mlflow.genai.evaluate()`` to run
scorers and ``result.assert_passed()`` to check the results.

    @mlflow.test
    def test_recommendations(agent):
        result = mlflow.genai.evaluate(
            predict_fn=agent.invoke,
            data=[{"inputs": {"prompt": "What tracing tool?"}}],
            scorers=[Guidelines(guidelines="Recommends MLflow"), Safety()],
        )
        result.assert_passed()
"""

from __future__ import annotations

from typing import Any, Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

MLFLOW_TEST_ATTR = "_mlflow_test"
MLFLOW_TEST_REPEAT_ATTR = "_mlflow_test_repeat"
MLFLOW_TEST_PASS_THRESHOLD_ATTR = "_mlflow_test_pass_threshold"


def test(
    fn: Callable[_P, _R] | None = None,
    *,
    repeat: int = 1,
    pass_threshold: int | None = None,
) -> Any:
    """Mark a test function for the MLflow pytest plugin.

    Supports both bare ``@mlflow.test`` and ``@mlflow.test(repeat=N)``.

    Args:
        fn: The test function (bare decorator) or ``None`` (called form).
        repeat: Re-run the test body N times; pass on a majority. Default 1.
        pass_threshold: How many of N runs must pass. Default: strict majority.

    Raises:
        ValueError: if ``repeat < 1`` or ``pass_threshold`` is out of range.
    """
    if repeat < 1:
        raise ValueError(f"@mlflow.test(repeat=...) must be >= 1, got {repeat}.")
    if pass_threshold is None:
        threshold = repeat // 2 + 1
    elif not 1 <= pass_threshold <= repeat:
        raise ValueError(
            f"@mlflow.test(pass_threshold=...) must be between 1 and repeat "
            f"({repeat}), got {pass_threshold}."
        )
    else:
        threshold = pass_threshold

    def mark(f: Callable[_P, _R]) -> Callable[_P, _R]:
        setattr(f, MLFLOW_TEST_ATTR, True)
        setattr(f, MLFLOW_TEST_REPEAT_ATTR, repeat)
        setattr(f, MLFLOW_TEST_PASS_THRESHOLD_ATTR, threshold)
        return f

    return mark if fn is None else mark(fn)
