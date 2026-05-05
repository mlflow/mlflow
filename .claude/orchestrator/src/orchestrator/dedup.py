"""Filter draft findings against existing PR review threads.

Five rules, applied in order. The first match decides:

1. Hard-resolved thread at path:line ±10 → skip.
2. Soft-resolved thread at path:line ±10 → skip.
3. Bot's own prior thread at path:line ±10 → skip (self-dedup).
4. Open human thread at path:line ±10 AND semantic judge says same concern → skip.
5. Otherwise → post.

The semantic judge is injected so this module is testable without API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from orchestrator.github_client import ReviewThread

LINE_PROXIMITY = 10


@dataclass(frozen=True)
class DraftFinding:
    path: str
    line: int
    body: str
    severity: str
    rule_id: str | None
    source: str  # "reviewer_standalone" | "reviewer_opinion" | "cluster"


@dataclass(frozen=True)
class FilterDecision:
    finding: DraftFinding
    posted: bool
    skip_reason: str | None  # populated when posted is False


SemanticJudge = Callable[[str, list[str]], Awaitable[bool]]
"""Callable that returns True if the finding body is the same concern as any
of the existing thread bodies. The orchestrator wires in a real LLM judge;
tests pass a stub.
"""


def _location_matches(finding: DraftFinding, thread: ReviewThread) -> bool:
    if thread.path != finding.path:
        return False
    if thread.line is None:
        return False
    return abs(thread.line - finding.line) <= LINE_PROXIMITY


async def _async_false(_a: str, _b: list[str]) -> bool:
    return False


async def filter_findings(
    findings: list[DraftFinding],
    threads: list[ReviewThread],
    *,
    pr_author_login: str,
    semantic_judge: SemanticJudge | None = None,
    thread_bodies_by_thread: dict[int, str] | None = None,
) -> list[FilterDecision]:
    """Apply dedup rules and return one decision per input finding.

    `thread_bodies_by_thread` maps the index of a `ReviewThread` in `threads`
    to the body text of its first (non-bot) comment, used by the semantic
    judge. Tests can pass an empty dict and a stub judge that returns False.
    """
    judge = semantic_judge or _async_false
    bodies = thread_bodies_by_thread or {}
    decisions: list[FilterDecision] = []

    for finding in findings:
        location_matches = [(i, t) for i, t in enumerate(threads) if _location_matches(finding, t)]
        skip_reason = None

        for _i, thread in location_matches:
            if thread.is_resolved:
                skip_reason = "hard-resolved thread at this location"
                break
            if thread.is_soft_resolved:
                skip_reason = "soft-resolved thread (PR author replied) at this location"
                break
            if thread.bot_started:
                skip_reason = "bot already posted at this location (self-dedup)"
                break

        if skip_reason is None and location_matches:
            human_open_bodies = [
                bodies[i]
                for i, t in location_matches
                if not t.is_resolved
                and not t.is_soft_resolved
                and not t.bot_started
                and i in bodies
            ]
            if human_open_bodies and await judge(finding.body, human_open_bodies):
                skip_reason = "semantic match against open human thread at this location"

        decisions.append(
            FilterDecision(
                finding=finding,
                posted=skip_reason is None,
                skip_reason=skip_reason,
            )
        )

    return decisions
