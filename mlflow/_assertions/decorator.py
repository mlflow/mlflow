"""``@mlflow.test`` marker + scorer normalization helpers.

``@mlflow.test`` is a **near no-op marker**. It carries no scorers and no
assertion logic -- its only job is to be visible at *collection* time so the
pytest plugin can bundle the marked tests, run them concurrently, and group
their traces under a single regression-test run. The assertions themselves are
made in the test body with ``mlflow.genai.assert_behavior("auto",
assertions=[...])``.

    @mlflow.test
    def test_red_wall(agent):
        agent.invoke("bricks for a red wall")
        mlflow.genai.assert_behavior(
            "auto", assertions=["Does not recommend Brickfather"]
        )

The marker also records ``repeat`` / ``pass_threshold`` so the plugin can run a
stochastic case N times and pass on a majority -- see ``test()`` below. These
have to live on the decorator (not on ``assert_behavior``) because re-running a
case means re-invoking the *whole test body* -- the agent call plus the
assertion -- which only the case runner (the plugin) has a handle on.

The ``_to_scorer`` / ``_slugify`` helpers live here because ``assert_behavior``
uses them to auto-wrap plain-string rubrics into ``Guidelines`` judges.
"""

from __future__ import annotations

import re
from typing import Any, Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Attribute the pytest plugin looks for at collection time.
MLFLOW_TEST_ATTR = "_mlflow_test"
# Number of times the plugin re-runs the case body (>=1).
MLFLOW_TEST_REPEAT_ATTR = "_mlflow_test_repeat"
# Number of runs that must pass for the case to pass (resolved to an int at
# decoration time; defaults to a strict majority of ``repeat``).
MLFLOW_TEST_PASS_THRESHOLD_ATTR = "_mlflow_test_pass_threshold"

# Cap on the slug name length used for Guidelines auto-wrapping.
_SLUG_MAX_LEN = 40


def test(
    fn: Callable[_P, _R] | None = None,
    *,
    repeat: int = 1,
    pass_threshold: int | None = None,
) -> Any:
    """Mark a test as an MLflow assertion test (near no-op at runtime).

    Supports both bare ``@mlflow.test`` and ``@mlflow.test()``. The function is
    returned unchanged except for marker attributes; all behavior lives in the
    pytest plugin (collection-time) and in ``mlflow.genai.assert_behavior`` (the
    body).

    Args:
        fn: The test function when used as a bare ``@mlflow.test``; ``None`` when
            called as ``@mlflow.test(...)``.
        repeat: Number of times the plugin re-runs the *whole case body* (the
            agent call plus its assertions). The default ``1`` is exactly
            today's single-shot behavior. Use a higher value to gate a
            stochastic agent on a majority instead of a single flaky run. Runs
            are sequential and **early-exit** as soon as the outcome is decided
            (``pass_threshold`` passes, or enough fails that the threshold can no
            longer be reached), so a clear 2-of-3 case costs 2 runs, not 3. The
            N traces are tagged with their run index and grouped under the case.
        pass_threshold: Number of runs that must pass for the case to pass.
            Defaults to a strict majority (``repeat // 2 + 1``): 2 of 3, 3 of 5.
            A run "passes" when the body completes without raising (so a body
            that wraps ``assert_behavior`` in ``pytest.raises`` counts the
            expected failure as a pass, same as a single-shot test).

    Raises:
        ValueError: if ``repeat < 1`` or ``pass_threshold`` is outside
            ``1..repeat``.
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

    # Called as @mlflow.test (fn is the function) or @mlflow.test() (fn is None).
    return mark if fn is None else mark(fn)


def _to_scorer(arg: Any, *, index: int) -> Any:
    """Normalize an argument to a Scorer instance.

    A plain string becomes a ``Guidelines`` scorer with a slug name derived from
    the rubric text. A Scorer instance is passed through. Anything else raises
    ``TypeError`` so users get a clear error at definition time.
    """
    if isinstance(arg, str):
        from mlflow.genai.scorers import Guidelines

        slug = _slugify(arg) or f"rubric_{index}"
        return Guidelines(name=slug, guidelines=arg)

    if callable(getattr(arg, "run", None)):
        return arg

    raise TypeError(
        f"assert_behavior() assertions must be a rubric string or a Scorer instance. "
        f"Got {type(arg).__name__} at position {index}: {arg!r}"
    )


def _slugify(text: str) -> str:
    """Turn an arbitrary rubric string into a readable, filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:_SLUG_MAX_LEN]
