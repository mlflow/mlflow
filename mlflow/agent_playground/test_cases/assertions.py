"""Evaluator for the ``assertion`` test-case strategy.

The agent_playground supports two test strategies: ``assertion`` and
``judge``. Assertion-strategy tests run a set of deterministic substring
and tool-call checks against the agent's final response, with no LLM
involved. This module owns the evaluator function for that strategy.

The :class:`AssertionExpectations` shape (``must_contain`` /
``must_not_contain`` / ``must_call_tool`` / ``must_not_call_tool``) is
the v1 PRD's deterministic-check contract and lives in
:mod:`mlflow.agent_playground.test_cases.entities` as the
``kind="assertion"`` variant of the :data:`Expectations` discriminated
union.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from mlflow.agent_playground.test_cases.entities import AssertionExpectations

# Narrower than the runner-level ``VerdictOutcome`` (which also admits
# ``"error"`` for execution-level failures). The assertion evaluator
# never produces ``"error"``; the runner wraps execution failures into
# a full ``Verdict`` directly. Encoding the contract here lets type
# checkers reject ``AssertionResult(outcome="error", ...)`` at
# construction sites instead of relying on a runtime guard.
AssertionOutcome: TypeAlias = Literal["pass", "fail"]


@dataclass(frozen=True)
class AssertionResult:
    """Outcome of evaluating an :class:`AssertionExpectations`.

    ``outcome`` is ``"pass"`` iff every clause held, ``"fail"`` if any
    clause failed. See :data:`AssertionOutcome` for why ``"error"`` is
    not part of the contract.

    ``reasons`` lists every failed clause as a human-readable string.
    The invariant is symmetric: ``outcome="pass"`` carries no reasons;
    ``outcome="fail"`` carries at least one. Mirrors
    :meth:`Verdict._validate_reasons_match_outcome` in
    :mod:`mlflow.agent_playground.test_cases.entities` with the
    intentional divergence that ``Verdict`` also permits ``"error"``
    while this dataclass does not. The eval-layer guard catches
    malformed callers at construction; the persisted ``Verdict`` guard
    catches the same shape after run-level wrapping.
    """

    outcome: AssertionOutcome
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome == "pass" and self.reasons:
            raise ValueError(f"outcome={self.outcome!r} must not carry reasons")
        if self.outcome == "fail" and not self.reasons:
            raise ValueError(f"outcome={self.outcome!r} must carry at least one reason")


def evaluate_assertions(
    spec: AssertionExpectations,
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
        spec: The assertion expectations from the test case row.
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
    "AssertionOutcome",
    "AssertionResult",
    "evaluate_assertions",
]
