"""``mlflow.genai.assert_behavior`` -- the imperative assertion API.

Runs scorers against the trace an agent produced, attaches each result to that
trace as feedback, records it for the session roll-up, and raises
``AssertionError`` if any scorer fails. Usable anywhere -- pytest, a notebook, or
a plain script. Under pytest, mark the test with ``@mlflow.test`` so the plugin
bundles and parallelizes it and groups its traces under the regression-test run.

The assertion is grounded in a *trace*, not a hand-passed input/output pair:
that is what separates it from ``mlflow.genai.evaluate`` (batch/dataset
oriented) and lets assertions cover intermediate behavior (tool calls,
retrieval) via trace-introspecting scorers, not just the final output.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

import mlflow
from mlflow._assertions import session
from mlflow._assertions.decorator import _to_scorer
from mlflow._assertions.runner import run_assertions
from mlflow.entities import Trace
from mlflow.genai.utils.trace_utils import (
    extract_inputs_from_trace,
    extract_outputs_from_trace,
)


def _resolve_identity(name: str | None) -> tuple[str | None, str | None]:
    """Figure out (test_name, case_id) for trace tagging.

    Prefers an explicit ``name=``; then the thread-local set by the bundle runner
    (inside a bundle, ``PYTEST_CURRENT_TEST`` names the bundle item, not the
    test); then parses ``PYTEST_CURRENT_TEST`` for the non-bundled case.
    """
    if name:
        return name, None
    tl_name, tl_case = session.current_test()
    if tl_name:
        return tl_name, tl_case
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if not current:
        return None, None
    nodeid = current.rsplit(" (", 1)[0]  # strip the " (call)" phase suffix
    item_name = nodeid.split("::")[-1]
    case_id = None
    if m := re.search(r"\[(.+)\]$", item_name):
        case_id = m.group(1)
        item_name = item_name[: m.start()]
    return item_name, case_id


def _resolve_trace(
    trace: Trace | Literal["auto"] | None,
) -> tuple[Trace | None, str | None]:
    """Resolve the ``trace`` argument to a ``(trace, trace_id)`` pair.

    ``None`` skips trace lookup entirely. ``"auto"`` looks up the last active
    trace produced in *this thread* -- so the bundle runner's parallel tests each
    resolve their own trace -- and materializes it, flushing any pending async
    export first. A ``Trace`` is used as-is.
    """
    if trace is None:
        return None, None
    if isinstance(trace, Trace):
        return trace, trace.info.trace_id
    if trace == "auto":
        trace_id = mlflow.get_last_active_trace_id(thread_local=True)
        trace_obj = mlflow.get_trace(trace_id, silent=True, flush=True) if trace_id else None
        return trace_obj, trace_id
    raise TypeError(
        'assert_behavior(trace=...) must be a Trace, the literal "auto", or None. '
        f"Got {type(trace).__name__}: {trace!r}"
    )


def assert_behavior(
    trace: Trace | Literal["auto"] | None = None,
    *,
    assertions: list[Any],
    inputs: Any = None,
    outputs: Any = None,
    name: str | None = None,
) -> None:
    """Assert that an agent's behavior on ``trace`` satisfies every assertion.

    Args:
        trace: The trace to score. Defaults to ``None`` (no trace lookup).
            Pass ``"auto"`` to use the last trace produced in the current thread,
            or an explicit :class:`~mlflow.entities.Trace`. Inputs and outputs
            are extracted from the trace unless overridden by
            ``inputs``/``outputs``.
        assertions: One or more assertions. Each is either a plain-string rubric
            (auto-wrapped in a ``Guidelines`` LLM judge) or a ``Scorer``
            instance (``Safety()``, ``Guidelines(...)``, a ``@scorer`` function).
        inputs: Optional inputs override, forwarded to assertions that declare
            them. Defaults to the inputs extracted from the trace.
        outputs: Optional outputs override, forwarded to assertions that declare
            them. Defaults to the outputs extracted from the trace. Pass this
            (with ``trace="auto"``) to score an output when nothing was traced.
        name: Optional test/case name for trace tagging. Defaults to the current
            pytest test when running under pytest.

    Raises:
        ValueError: if ``assertions`` is empty.
        TypeError: if ``trace`` is not a ``Trace`` or the literal ``"auto"``.
        AssertionError: if any assertion reports a failing value.
    """
    if not assertions:
        raise ValueError(
            "assert_behavior(...) requires at least one assertion. Pass a rubric "
            "string like 'Refuses politely', a scorer instance like Safety(), or "
            "a @scorer-decorated function."
        )

    resolved = [_to_scorer(s, index=i) for i, s in enumerate(assertions)]
    trace_obj, trace_id = _resolve_trace(trace)

    # Extract inputs/outputs from the trace when the caller did not override them,
    # so scorers that declare `inputs`/`outputs` (e.g. a plain @scorer function)
    # receive them without the user repeating what the trace already records.
    if trace_obj is not None:
        if inputs is None:
            inputs = extract_inputs_from_trace(trace_obj)
        if outputs is None:
            outputs = extract_outputs_from_trace(trace_obj)

    test_name, case_id = _resolve_identity(name)

    # Open the regression-test run (once) before scoring so this and subsequent
    # traces link to it. Idempotent; non-fatal if tracking is unavailable.
    session.ensure_run()
    # repeat_index is set by the plugin only while running a repeated case, so
    # each of the N traces is tagged with its run index and grouped under the case.
    trace_tags = session.build_trace_tags(test_name, case_id, session.repeat_index())

    results = run_assertions(
        resolved,
        inputs=inputs,
        outputs=outputs,
        trace=trace_obj,
        trace_id=trace_id,
        trace_tags=trace_tags,
    )
    session.record(test_name or "<assert_behavior>", results)

    failures = [r for r in results if not r.passed]
    if not failures:
        return

    # Pack the actionable signal onto the first line so pytest's short summary
    # tells the user *why* it failed, not just that assert_behavior() raised.
    if len(failures) == 1:
        raise AssertionError(failures[0].summary())
    names = ", ".join(r.scorer_name for r in failures)
    detail = "\n".join(f"  - {r.summary()}" for r in failures)
    raise AssertionError(f"{len(failures)} assertions failed: {names}\n{detail}")
