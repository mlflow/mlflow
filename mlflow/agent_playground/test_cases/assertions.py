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

from collections.abc import Iterable
from dataclasses import dataclass

from mlflow.agent_playground.test_cases.entities import AssertionSpec, VerdictOutcome


@dataclass(frozen=True)
class AssertionResult:
    """Outcome of evaluating an :class:`AssertionSpec`.

    ``outcome`` is ``"pass"`` iff every clause held, ``"fail"`` if any
    clause failed. The assertion evaluator never produces ``"error"``;
    execution-level failures (agent crash, timeout) are surfaced by the
    runner when it wraps this into a full
    :class:`mlflow.agent_playground.test_cases.entities.Verdict` with
    run-level context (``test_case_id``, ``trace_ids``,
    ``duration_ms``).

    ``reasons`` lists every failed clause as a human-readable string;
    it is empty when ``outcome == "pass"``.
    """

    outcome: VerdictOutcome
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome == "pass" and self.reasons:
            raise ValueError("outcome='pass' must not carry reasons")


def evaluate_assertions(
    spec: AssertionSpec,
    final_response_text: str,
    tool_call_names: Iterable[str],
) -> AssertionResult:
    """Apply ``spec`` to the agent's final response and tool-call list.

    Substring checks (``must_contain`` / ``must_not_contain``) are
    case-sensitive and do not normalize Unicode; needle strings match
    byte-for-byte against ``final_response_text``. Tool-call checks
    treat ``tool_call_names`` as a set: ``must_call_tool=["foo"]`` holds
    if ``"foo"`` appears at least once, regardless of how many times it
    was invoked.

    Args:
        spec: The assertion spec from the test case row.
        final_response_text: The assistant's text in the final turn of
            the conversation.
        tool_call_names: Names of tool spans observed during the
            conversation (any turn, not just the final one). Membership
            tested with set semantics; multiplicity and order ignored.

    Returns:
        :class:`AssertionResult` with ``outcome`` and a ``reasons``
        tuple listing each failed clause.
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
    return AssertionResult(
        outcome="pass" if not reasons else "fail",
        reasons=tuple(reasons),
    )


__all__ = [
    "AssertionResult",
    "evaluate_assertions",
]
