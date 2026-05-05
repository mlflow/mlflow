from __future__ import annotations

import asyncio

import pytest
from orchestrator.dedup import DraftFinding, filter_findings
from orchestrator.github_client import ReviewThread


def _draft(path: str = "f.py", line: int = 100, body: str = "concern A") -> DraftFinding:
    return DraftFinding(
        path=path,
        line=line,
        body=body,
        severity="ask",
        rule_id=None,
        source="cluster",
    )


def _thread(
    path: str = "f.py",
    line: int | None = 100,
    is_resolved: bool = False,
    is_soft_resolved: bool = False,
    bot_started: bool = False,
    comment_authors: tuple[str, ...] = ("alice",),
) -> ReviewThread:
    return ReviewThread(
        path=path,
        line=line,
        is_resolved=is_resolved,
        comment_authors=comment_authors,
        is_soft_resolved=is_soft_resolved,
        bot_started=bot_started,
    )


async def _judge_false(_a: str, _b: list[str]) -> bool:
    return False


async def _judge_true(_a: str, _b: list[str]) -> bool:
    return True


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.parametrize(
    ("thread_kwargs", "expected_skip_substring"),
    [
        ({"is_resolved": True}, "hard-resolved"),
        ({"is_soft_resolved": True}, "soft-resolved"),
        ({"bot_started": True}, "self-dedup"),
    ],
)
def test_each_skip_rule_filters_finding(
    thread_kwargs: dict[str, bool], expected_skip_substring: str
) -> None:
    finding = _draft()
    threads = [_thread(**thread_kwargs)]
    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=_judge_false,
        )
    )
    assert len(decisions) == 1
    assert decisions[0].posted is False
    assert decisions[0].skip_reason is not None
    assert expected_skip_substring in decisions[0].skip_reason


def test_open_human_thread_with_no_semantic_match_lets_post() -> None:
    finding = _draft()
    threads = [_thread()]
    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=_judge_false,
            thread_bodies_by_thread={0: "different concern"},
        )
    )
    assert decisions[0].posted is True
    assert decisions[0].skip_reason is None


def test_open_human_thread_with_semantic_match_skips() -> None:
    finding = _draft()
    threads = [_thread()]
    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=_judge_true,
            thread_bodies_by_thread={0: "same concern as draft"},
        )
    )
    assert decisions[0].posted is False
    assert decisions[0].skip_reason == "semantic match against open human thread at this location"


def test_no_threads_at_location_lets_post() -> None:
    finding = _draft(line=100)
    threads = [_thread(line=200)]
    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=_judge_false,
        )
    )
    assert decisions[0].posted is True


@pytest.mark.parametrize(
    ("draft_line", "thread_line", "expected_posted"),
    [
        (100, 100, False),  # exact match
        (100, 109, False),  # within ±10 (9 below)
        (100, 110, False),  # within ±10 (10 below, boundary)
        (100, 111, True),  # outside ±10
        (100, 90, False),  # within ±10 (10 above, boundary)
        (100, 89, True),  # outside ±10
    ],
)
def test_line_proximity_window(draft_line: int, thread_line: int, expected_posted: bool) -> None:
    finding = _draft(line=draft_line)
    threads = [_thread(line=thread_line, is_resolved=True)]
    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=_judge_false,
        )
    )
    assert decisions[0].posted is expected_posted


def test_thread_line_none_is_skipped_in_match() -> None:
    finding = _draft(line=100)
    threads = [_thread(line=None, is_resolved=True)]
    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=_judge_false,
        )
    )
    assert decisions[0].posted is True


def test_resolved_takes_precedence_over_semantic_judge() -> None:
    finding = _draft()
    threads = [_thread(is_resolved=True)]
    judge_calls: list[tuple[str, list[str]]] = []

    async def tracking_judge(body: str, threads: list[str]) -> bool:
        judge_calls.append((body, threads))
        return True

    decisions = asyncio.run(
        filter_findings(
            [finding],
            threads,
            pr_author_login="someone",
            semantic_judge=tracking_judge,
            thread_bodies_by_thread={0: "any body"},
        )
    )
    assert decisions[0].posted is False
    assert "hard-resolved" in decisions[0].skip_reason
    assert judge_calls == []  # judge was not called
