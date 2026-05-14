"""AssertionSpec evaluator for the assertion test strategy.

The agent_playground supports two test strategies: ``assertion`` and
``judge``. Assertion-strategy tests run a set of deterministic substring
and tool-call checks against the agent's final response, with no LLM
involved. This module owns the evaluator function for that strategy.

The :class:`AssertionSpec` shape (``must_contain`` / ``must_not_contain``
/ ``must_call_tool`` / ``must_not_call_tool``) is the v1 PRD's
deterministic-check contract and lives in
:mod:`mlflow.agent_playground.test_cases.entities`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlflow.agent_playground.test_cases.entities import AssertionSpec


@dataclass(frozen=True)
class AssertionResult:
    """Outcome of evaluating an :class:`AssertionSpec`.

    ``passed`` is true iff every clause held. ``reasons`` lists every
    failed clause as a human-readable string; it is empty when
    ``passed`` is true. The runner wraps this into a full
    :class:`mlflow.agent_playground.test_cases.entities.Verdict` with
    run-level context (test_case_id, trace_ids, duration_ms).
    """

    passed: bool
    reasons: tuple[str, ...] = ()


def evaluate_assertions(
    spec: AssertionSpec,
    final_response_text: str,
    tool_call_names: list[str],
) -> AssertionResult:
    """Apply ``spec`` to the agent's final response and tool-call list.

    Args:
        spec: The assertion spec from the test case row.
        final_response_text: The assistant's text in the final turn of
            the conversation. Substring checks run against this.
        tool_call_names: Names of tool spans observed during the
            conversation (any turn, not just the final one). Tool-call
            checks run against this set.

    Returns:
        :class:`AssertionResult` with ``passed`` and a ``reasons`` tuple
        listing each failed clause.
    """
    reasons: list[str] = [
        f"must_contain: {needle!r} was not present in the response"
        for needle in spec.must_contain
        if needle not in final_response_text
    ]
    reasons.extend(
        f"must_not_contain: {needle!r} was present in the response"
        for needle in spec.must_not_contain
        if needle in final_response_text
    )
    called = set(tool_call_names)
    reasons.extend(
        f"must_call_tool: {tool!r} was not invoked"
        for tool in spec.must_call_tool
        if tool not in called
    )
    reasons.extend(
        f"must_not_call_tool: {tool!r} was invoked"
        for tool in spec.must_not_call_tool
        if tool in called
    )
    return AssertionResult(passed=not reasons, reasons=tuple(reasons))


__all__ = [
    "AssertionResult",
    "evaluate_assertions",
]
