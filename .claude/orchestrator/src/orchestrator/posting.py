"""Post bot review comments to a PR.

Two posting paths:

- One inline review comment per cluster (anchored at path:line on the PR's
  head SHA, side=RIGHT).
- One top-level summary comment with a hidden HTML marker containing the
  head SHA. The marker enables `--no-cache` cache-hit detection: a future
  /review on the same SHA scans bot summary comments, finds the marker,
  and early-exits.

Posting goes through `gh api` so the bot's GitHub App installation token
(provided as `GH_TOKEN`) is used; comments appear authored by
`mlflow-reviewer[bot]`.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

from orchestrator.dedup import DraftFinding
from orchestrator.github_client import PRMetadata

_logger = logging.getLogger(__name__)

POST_TRAILER = "Posted by `mlflow-reviewer` (auto-generated)."


@dataclass(frozen=True)
class PostedComment:
    finding_id: str | None
    html_url: str


def _format_inline_body(finding: DraftFinding) -> str:
    return f"{finding.body}\n\n---\n{POST_TRAILER}"


def _format_summary_body(
    pr: PRMetadata,
    posted_count: int,
    skipped_count: int,
    mode: str,
) -> str:
    return (
        f"`mlflow-reviewer` ran in `{mode}` mode against this PR.\n\n"
        f"- Posted: {posted_count} inline comment(s)\n"
        f"- Skipped: {skipped_count} (already covered by an existing thread or "
        "self-dedup)\n\n"
        f"Reply on any inline thread to discuss. Re-run with `/review --no-cache` "
        "to force a fresh review.\n\n"
        f"---\n{POST_TRAILER}\n\n"
        f"<!-- mlflow-reviewer: head_sha={pr.head_sha} -->"
    )


def _format_zero_findings_body(pr: PRMetadata, mode: str) -> str:
    return (
        f"`mlflow-reviewer` ran in `{mode}` mode against this PR and found "
        "no concerns to surface.\n\n"
        f"---\n{POST_TRAILER}\n\n"
        f"<!-- mlflow-reviewer: head_sha={pr.head_sha} -->"
    )


def post_inline_comment(
    repo: str,
    pr: int,
    head_sha: str,
    finding: DraftFinding,
) -> PostedComment:
    """Post one inline PR review comment via `gh api`.

    Uses POST /repos/{owner}/{repo}/pulls/{pr}/comments. Anchors at
    finding.path:finding.line on the right (post-PR) side. The current head
    SHA is required to ensure the comment lands at the line the orchestrator
    reviewed.
    """
    body = _format_inline_body(finding)
    args = [
        "gh",
        "api",
        f"repos/{repo}/pulls/{pr}/comments",
        "-X",
        "POST",
        "-f",
        f"body={body}",
        "-f",
        f"path={finding.path}",
        "-F",
        f"line={finding.line}",
        "-f",
        "side=RIGHT",
        "-f",
        f"commit_id={head_sha}",
    ]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    response = json.loads(result.stdout)
    return PostedComment(finding_id=None, html_url=response.get("html_url", ""))


def post_summary_comment(
    repo: str,
    pr: PRMetadata,
    *,
    posted_count: int,
    skipped_count: int,
    mode: str,
) -> PostedComment:
    """Post the PR-level summary as an issue comment.

    GitHub treats PR-level conversation comments as issue comments, so we
    POST /repos/{owner}/{repo}/issues/{pr}/comments.
    """
    body = (
        _format_zero_findings_body(pr, mode)
        if posted_count == 0
        else _format_summary_body(pr, posted_count, skipped_count, mode)
    )
    args = [
        "gh",
        "api",
        f"repos/{repo}/issues/{pr.number}/comments",
        "-X",
        "POST",
        "-f",
        f"body={body}",
    ]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    response = json.loads(result.stdout)
    return PostedComment(finding_id=None, html_url=response.get("html_url", ""))


def already_reviewed_at_sha(
    repo: str,
    pr: int,
    head_sha: str,
    bot_login: str = "mlflow-reviewer[bot]",
) -> bool:
    """Check whether the bot already posted a summary at the current head SHA.

    Scans issue comments authored by the bot for a hidden marker:
    `<!-- mlflow-reviewer: head_sha=<sha> -->`.

    Returns True if a matching marker is found, False otherwise.
    """
    args = [
        "gh",
        "api",
        f"repos/{repo}/issues/{pr}/comments",
        "--paginate",
        "-q",
        f'.[] | select(.user.login == "{bot_login}") | .body',
    ]
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        _logger.warning("Could not check prior bot comments: %s", e)
        return False
    marker = f"<!-- mlflow-reviewer: head_sha={head_sha} -->"
    return marker in result.stdout


def react_eyes_on_trigger(
    repo: str,
    comment_id: int,
) -> None:
    """Add an `eyes` reaction to the trigger comment so the maintainer sees
    the bot picked up the request.
    """
    args = [
        "gh",
        "api",
        f"repos/{repo}/issues/comments/{comment_id}/reactions",
        "-X",
        "POST",
        "-f",
        "content=eyes",
    ]
    subprocess.run(args, check=False, capture_output=True, text=True)


def react_completion_on_trigger(
    repo: str,
    comment_id: int,
    *,
    success: bool,
) -> None:
    """Add `+1` or `-1` reaction once the run completes."""
    content = "+1" if success else "-1"
    args = [
        "gh",
        "api",
        f"repos/{repo}/issues/comments/{comment_id}/reactions",
        "-X",
        "POST",
        "-f",
        f"content={content}",
    ]
    subprocess.run(args, check=False, capture_output=True, text=True)


def post_failure_comment(
    repo: str,
    pr: int,
    *,
    error_summary: str,
    run_url: str | None,
) -> None:
    """Post a brief failure comment when the orchestrator errors out."""
    run_link = f"\n\nSee [run logs]({run_url}) for details." if run_url else ""
    body = f"`mlflow-reviewer` failed: {error_summary}{run_link}\n\n---\n{POST_TRAILER}"
    args = [
        "gh",
        "api",
        f"repos/{repo}/issues/{pr}/comments",
        "-X",
        "POST",
        "-f",
        f"body={body}",
    ]
    subprocess.run(args, check=False, capture_output=True, text=True)
