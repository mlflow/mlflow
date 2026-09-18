"""Publish a validated issue-reproduction triage report.

This module intentionally contains no model or reproduction logic.  It is used
by the write-only workflow job after that job has fetched the current issue.
"""

import argparse
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER = "<!-- issue-repro-triage:v1 -->"
OUTCOMES = frozenset({"ready", "needs_committer_feedback"})
LABELS = {
    "ready": "ready",
    "needs_committer_feedback": "needs committer feedback",
}


@dataclass(frozen=True)
class PublishPlan:
    """The small, deterministic set of mutations allowed by this publisher."""

    markdown: str
    outcome: str
    comment_id: int | None
    add_label: str | None
    remove_label: str | None


def validate_report(report: Mapping[str, Any], repository: str, issue_number: int) -> None:
    """Reject reports that are not bound to this repository and issue."""
    binding = report.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("report is missing its binding")
    if binding.get("repository") != repository or binding.get("issue_number") != issue_number:
        raise ValueError("report binding does not match the requested issue")
    if report.get("outcome") not in OUTCOMES:
        raise ValueError("report has an invalid outcome")
    markdown = report.get("markdown")
    if not isinstance(markdown, str) or not markdown.startswith(MARKER):
        raise ValueError("report has no trusted triage marker")
    if len(markdown) > 20_000:
        raise ValueError("report markdown is too large")


def validate_issue(issue: Mapping[str, Any], issue_number: int) -> None:
    """Only publish against the still-open issue that was requested."""
    if issue.get("number") != issue_number:
        raise ValueError("fetched issue number does not match")
    if issue.get("state") != "open":
        raise ValueError("will not publish to a closed issue")
    if "pull_request" in issue:
        raise ValueError("will not publish to a pull request")


def build_publish_plan(
    report: Mapping[str, Any],
    issue: Mapping[str, Any],
    comments: list[Mapping[str, Any]],
    *,
    repository: str,
    issue_number: int,
    bot_login: str,
) -> PublishPlan:
    """Revalidate input and calculate only the permitted issue mutations."""
    validate_report(report, repository, issue_number)
    validate_issue(issue, issue_number)
    if not bot_login:
        raise ValueError("bot login is required")

    marker_comments = [
        comment
        for comment in comments
        if isinstance(comment.get("body"), str)
        and MARKER in comment["body"]
        and isinstance(comment.get("user"), Mapping)
        and comment["user"].get("login") == bot_login
        and isinstance(comment.get("id"), int)
    ]
    # The API is ordered oldest-first. Updating the newest bot-owned report
    # keeps a legacy duplicate untouched instead of editing user content.
    comment_id = marker_comments[-1]["id"] if marker_comments else None

    current_labels = {
        label.get("name")
        for label in issue.get("labels", [])
        if isinstance(label, Mapping) and isinstance(label.get("name"), str)
    }
    outcome = report["outcome"]
    add_label = LABELS[outcome] if LABELS[outcome] not in current_labels else None
    opposite = LABELS["needs_committer_feedback" if outcome == "ready" else "ready"]
    remove_label = opposite if opposite in current_labels else None
    return PublishPlan(
        markdown=report["markdown"],
        outcome=outcome,
        comment_id=comment_id,
        add_label=add_label,
        remove_label=remove_label,
    )


class GitHubClient:
    """Minimal GitHub API adapter; methods are deliberately easy to fake in tests."""

    def __init__(self, run: Callable[..., Any] = subprocess.run):
        self._run = run

    def api(self, endpoint: str, *, method: str = "GET", body: Any = None) -> Any:
        command = ["gh", "api", "--method", method, endpoint]
        kwargs: dict[str, Any] = {"check": True, "capture_output": True, "text": True}
        if body is not None:
            command.extend(["--input", "-"])
            kwargs["input"] = json.dumps(body)
        completed = self._run(command, **kwargs)
        return json.loads(completed.stdout) if completed.stdout else None

    def get_issue(self, repository: str, issue_number: int) -> Mapping[str, Any]:
        issue = self.api(f"repos/{repository}/issues/{issue_number}")
        if not isinstance(issue, Mapping):
            raise ValueError("GitHub returned an invalid issue")
        return issue

    def get_comments(self, repository: str, issue_number: int) -> list[Mapping[str, Any]]:
        comments: list[Mapping[str, Any]] = []
        page = 1
        while True:
            result = self.api(
                f"repos/{repository}/issues/{issue_number}/comments?per_page=100&page={page}"
            )
            if not isinstance(result, list):
                raise ValueError("GitHub returned invalid comments")
            if not all(isinstance(comment, Mapping) for comment in result):
                raise ValueError("GitHub returned an invalid comment")
            comments.extend(result)
            if len(result) < 100:
                return comments
            page += 1

    def apply(self, repository: str, issue_number: int, plan: PublishPlan) -> None:
        endpoint = f"repos/{repository}/issues/{issue_number}"
        if plan.comment_id is None:
            self.api(f"{endpoint}/comments", method="POST", body={"body": plan.markdown})
        else:
            self.api(
                f"repos/{repository}/issues/comments/{plan.comment_id}",
                method="PATCH",
                body={"body": plan.markdown},
            )
        if plan.remove_label:
            self.api(f"{endpoint}/labels/{plan.remove_label}", method="DELETE")
        if plan.add_label:
            self.api(f"{endpoint}/labels", method="POST", body={"labels": [plan.add_label]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--bot-login", required=True)
    parser.add_argument("--apply-changes", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.issue_number < 1:
        parser.error("--issue-number must be positive")

    report = json.loads(args.report.read_text())
    client = GitHubClient()
    plan = build_publish_plan(
        report,
        client.get_issue(args.repository, args.issue_number),
        client.get_comments(args.repository, args.issue_number),
        repository=args.repository,
        issue_number=args.issue_number,
        bot_login=args.bot_login,
    )
    if args.summary:
        args.summary.write_text(plan.markdown + "\n")
    if args.apply_changes:
        client.apply(args.repository, args.issue_number, plan)


if __name__ == "__main__":
    main()
