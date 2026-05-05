"""GitHub PR access for the orchestrator.

Reads PR metadata, diff, and existing review threads. Posting is in `posting.py`
(Stack 3) so that read and write paths can be reviewed independently.

Stack 1 ships the data shapes the rest of the orchestrator will consume. The
actual gh-CLI / GraphQL calls land in Stack 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRMetadata:
    """Subset of `gh pr view` data the orchestrator needs."""

    number: int
    title: str
    author_login: str
    head_sha: str
    base_ref: str
    labels: tuple[str, ...]
    changed_paths: tuple[str, ...]
    additions: int
    deletions: int


@dataclass(frozen=True)
class ReviewThread:
    """One PR review thread, used by dedup logic.

    Soft-resolved: `is_resolved is False` AND `pr_author_replied is True` AND
    a non-bot non-author started the thread. The dedup module computes that
    flag from these fields plus the PR author's login.
    """

    path: str
    line: int | None
    is_resolved: bool
    comment_authors: tuple[str, ...]
    pr_author_replied: bool


# Bot logins the dedup module should ignore when computing "is this a human
# thread" or "did a human reply": bot replies don't count as acknowledgment.
BOT_LOGINS_TO_IGNORE: frozenset[str] = frozenset({
    "copilot-pull-request-reviewer",
    "github-actions",
    "mlflow-app",
    "mlflow-automation",
    "dependabot",
    "pre-commit-ci",
})
