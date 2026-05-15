"""Hard-match deduplication for agent_playground test cases.

When a new test case is being inserted by the test-gen worker, this module
runs the fast hard-match check before the worker invokes the connected
coding agent for the more expensive semantic-similarity dedup pass.

The hard match catches the common case: a user re-clicks feedback on the
same assistant message. Same anchored message means same complaint,
regardless of wording. No LLM call needed; this is a pure function over
the existing row metadata.

For the slower semantic check (paraphrased complaints on different
messages), the test-gen worker invokes the CoderAdapter directly. That
lives in the worker rather than here because it requires the connected
coding agent and this module deliberately stays pure / no-IO.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

HardMatchReason: TypeAlias = Literal["anchored_message_id"]


@dataclass(frozen=True)
class HardMatchUnique:
    """The new case has no hard-match against any existing case."""


@dataclass(frozen=True)
class HardMatchDuplicate:
    """The new case duplicates an existing one by hard match.

    ``existing_test_case_id`` names the duplicate-of pointer (matches
    the :class:`mlflow.agent_playground.test_cases.entities.DedupVerdict`
    field of the same name).

    ``reason`` is currently always ``"anchored_message_id"`` and is kept
    as a discriminator so future hard-match rules (e.g., identical
    conversation prefix) can be added without breaking callers.
    """

    existing_test_case_id: str
    reason: HardMatchReason = "anchored_message_id"


HardMatchResult: TypeAlias = HardMatchUnique | HardMatchDuplicate


def check_hard_match(
    new_source_assistant_message_id: str | None,
    existing: Iterable[tuple[str | None, str]],
) -> HardMatchResult:
    """Check whether any existing case is anchored on the same message.

    Args:
        new_source_assistant_message_id: Anchored message id of the new
            case, or ``None`` if the new case has no anchor.
        existing: Iterable of ``(source_assistant_message_id,
            test_case_id)`` tuples drawn from existing rows in the
            experiment's regression dataset.

    Returns:
        :class:`HardMatchDuplicate` when an existing row carries the
        same anchored message id; :class:`HardMatchUnique` otherwise.
        When the new case has no anchor, always returns
        :class:`HardMatchUnique` (no hard match is possible without an
        anchor).

    The first matching existing row is returned; later rows are not
    consulted. The caller invokes the coder-mediated semantic dedup pass
    when this function returns :class:`HardMatchUnique`.
    """
    if new_source_assistant_message_id is None:
        return HardMatchUnique()
    return next(
        (
            HardMatchDuplicate(existing_test_case_id=existing_id)
            for existing_msg_id, existing_id in existing
            if existing_msg_id == new_source_assistant_message_id
        ),
        HardMatchUnique(),
    )


__all__ = [
    "HardMatchDuplicate",
    "HardMatchReason",
    "HardMatchResult",
    "HardMatchUnique",
    "check_hard_match",
]
