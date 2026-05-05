"""GitHub PR access for the orchestrator.

Reads PR metadata, diff, file contents at head SHA, and existing review
threads. Posting is in `posting.py` (Stack 3) so read and write paths can be
reviewed independently.

Uses the `gh` CLI as a subprocess. The CLI is preinstalled on
GitHub-hosted Actions runners, and `gh auth` picks up `GH_TOKEN` from the
environment so workflow steps can pass an installation token directly.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PRMetadata:
    number: int
    title: str
    body: str
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

    `is_soft_resolved` is computed at fetch time from `is_resolved`,
    `pr_author_replied`, and `non_author_started`.
    """

    path: str
    line: int | None
    is_resolved: bool
    comment_authors: tuple[str, ...]
    is_soft_resolved: bool
    bot_started: bool


# Bot logins that should not count as "human reply" or "human acknowledgment".
BOT_LOGINS_TO_IGNORE: frozenset[str] = frozenset({
    "copilot-pull-request-reviewer",
    "github-actions",
    "mlflow-app",
    "mlflow-automation",
    "dependabot",
    "pre-commit-ci",
})


class GitHubClient:
    """Wrapper around `gh` CLI subprocess calls.

    All methods raise `subprocess.CalledProcessError` on non-zero exit. Callers
    decide whether to retry or surface the error to the workflow log.
    """

    def __init__(
        self, repo: str = "mlflow/mlflow", bot_login: str = "mlflow-reviewer[bot]"
    ) -> None:
        self._repo = repo
        self._bot_login = bot_login

    def get_pr_metadata(self, pr: int) -> PRMetadata:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr),
                "--repo",
                self._repo,
                "--json",
                "number,title,body,labels,baseRefName,headRefOid,additions,deletions,files,author",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        return PRMetadata(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            author_login=data["author"]["login"],
            head_sha=data["headRefOid"],
            base_ref=data["baseRefName"],
            labels=tuple(label["name"] for label in data.get("labels", [])),
            changed_paths=tuple(f["path"] for f in data.get("files", [])),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
        )

    def get_pr_diff(self, pr: int) -> str:
        result = subprocess.run(
            ["gh", "pr", "diff", str(pr), "--repo", self._repo],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def read_file_at_head(self, path: str, head_sha: str) -> str:
        """Read a file's contents at the PR head SHA via `git show`.

        Requires the workflow to have checked out the PR head ref into the
        working tree first (the workflow does this with
        `actions/checkout@v4`).
        """
        result = subprocess.run(
            ["git", "show", f"{head_sha}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def get_review_threads(self, pr: int, pr_author: str) -> list[ReviewThread]:
        """Return all review threads on the PR with computed soft-resolved flag.

        Soft-resolved: `is_resolved=False` AND the PR author has replied to a
        thread started by a non-author non-bot. The flag is computed here so
        the dedup module does not have to know about author/bot semantics.
        """
        query = """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              reviewThreads(first: 100) {
                nodes {
                  isResolved
                  path
                  line
                  originalLine
                  startLine
                  comments(first: 20) {
                    nodes {
                      author { login }
                    }
                  }
                }
              }
            }
          }
        }
        """
        owner, name = self._repo.split("/")
        result = subprocess.run(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={query}",
                "-F",
                f"owner={owner}",
                "-F",
                f"name={name}",
                "-F",
                f"number={pr}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        match json.loads(result.stdout):
            case {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}}:
                pass
            case _:
                _logger.warning("Unexpected GraphQL shape from review threads query.")
                return []

        threads: list[ReviewThread] = []
        for node in nodes:
            line = node.get("line") or node.get("originalLine") or node.get("startLine")
            authors = tuple(
                c["author"]["login"]
                for c in node.get("comments", {}).get("nodes", [])
                if c.get("author")
            )
            non_bot_authors = [a for a in authors if a not in BOT_LOGINS_TO_IGNORE]
            opener = authors[0] if authors else None
            bot_started = opener == self._bot_login
            non_author_opener = (
                opener is not None and opener != pr_author and opener not in BOT_LOGINS_TO_IGNORE
            )
            pr_author_replied = pr_author in authors[1:] if len(authors) > 1 else False
            is_resolved = node.get("isResolved", False) and bool(non_bot_authors)
            is_soft_resolved = (
                not node.get("isResolved", False) and non_author_opener and pr_author_replied
            )
            threads.append(
                ReviewThread(
                    path=node.get("path") or "",
                    line=line,
                    is_resolved=is_resolved,
                    comment_authors=authors,
                    is_soft_resolved=is_soft_resolved,
                    bot_started=bot_started,
                )
            )
        return threads
