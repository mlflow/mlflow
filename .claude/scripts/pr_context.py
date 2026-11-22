"""Print a pull request's context as JSON.

The output describes the PR and has a timeline that mirrors the Conversation tab: comments and
reviews in chronological order, with each inline thread under the review that started it.

Usage:
    uv run --no-project .claude/scripts/pr_context.py --repo mlflow/mlflow --pr-number 123
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $endCursor) {
        nodes {
          id path line originalLine isResolved isOutdated
          comments(first: 1) { nodes { id } }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def _api(*args: str) -> Any:
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def _list_items(endpoint: str) -> list[dict[str, Any]]:
    return [item for page in _api("--paginate", "--slurp", endpoint) for item in page]


def _list_threads(repo: str, number: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/")
    pages = _api(
        "graphql",
        "--paginate",
        "--slurp",
        "-f",
        f"query={_THREADS_QUERY}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={name}",
        "-F",
        f"number={number}",
    )
    return [
        thread
        for page in pages
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    ]


def _login(item: dict[str, Any]) -> str:
    return (item["user"] or {}).get("login", "")


def describe_pr(pr: dict[str, Any]) -> dict[str, Any]:
    user = pr["user"] or {}
    return {
        "number": pr["number"],
        "url": pr["html_url"],
        "title": pr["title"],
        "body": pr["body"] or "",
        "state": pr["state"],
        "draft": pr["draft"],
        "author": {
            "login": user.get("login", ""),
            "type": user.get("type", ""),
            "association": pr["author_association"],
        },
        "base": {
            "ref": pr["base"]["ref"],
            "sha": pr["base"]["sha"],
        },
        "head": {
            "ref": pr["head"]["ref"],
            "sha": pr["head"]["sha"],
            # The head repository is gone once a fork is deleted.
            "repo": (pr["head"]["repo"] or {}).get("full_name", ""),
        },
    }


def build_timeline(
    *,
    issue_comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
    threads: list[dict[str, Any]],
    review_id: int | None = None,
) -> list[dict[str, Any]]:
    by_root: dict[int, list[dict[str, Any]]] = {}
    for comment in review_comments:
        by_root.setdefault(comment.get("in_reply_to_id") or comment["id"], []).append(comment)
    roots = {c["node_id"]: c["id"] for c in review_comments}
    started: dict[int, list[dict[str, Any]]] = {}
    for thread in threads:
        root_nodes = thread["comments"]["nodes"]
        if not root_nodes or root_nodes[0]["id"] not in roots:
            continue
        conversation = by_root[roots[root_nodes[0]["id"]]]
        started.setdefault(conversation[0]["pull_request_review_id"], []).append({
            "id": thread["id"],
            "path": thread["path"],
            "line": thread["line"] or thread["originalLine"],
            "resolved": thread["isResolved"],
            "outdated": thread["isOutdated"],
            "diff_hunk": conversation[0]["diff_hunk"],
            "comments": [
                {
                    "id": c["id"],
                    "author": _login(c),
                    "created_at": c["created_at"],
                    "body": c["body"],
                    "review_id": c["pull_request_review_id"],
                }
                for c in conversation
            ],
        })
    timeline = [
        {
            "type": "comment",
            "author": _login(c),
            "created_at": c["created_at"],
            "body": c["body"],
        }
        for c in issue_comments
    ] + [
        {
            "type": "review",
            "id": r["id"],
            "author": _login(r),
            "state": r["state"],
            "created_at": r["submitted_at"],
            "body": r["body"] or "",
            "url": r["html_url"],
            "threads": started.get(r["id"], []),
        }
        for r in reviews
        # Replies made with "Add single comment" create empty reviews that GitHub doesn't
        # show; their comments already appear in the threads they belong to.
        if r["id"] == review_id or r["body"] or r["state"] != "COMMENTED" or r["id"] in started
    ]
    timeline.sort(key=lambda item: item["created_at"])
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a pull request's description and conversation timeline as JSON."
    )
    parser.add_argument("--repo", required=True, help="Repository as <owner>/<name>")
    parser.add_argument("--pr-number", type=int, required=True, help="Pull request number")
    parser.add_argument(
        "--review-id",
        type=int,
        help="Review to keep even if it is empty, which GitHub would otherwise hide",
    )
    args = parser.parse_args()

    endpoint = f"repos/{args.repo}/pulls/{args.pr_number}"
    timeline = build_timeline(
        issue_comments=_list_items(f"repos/{args.repo}/issues/{args.pr_number}/comments"),
        reviews=_list_items(f"{endpoint}/reviews"),
        review_comments=_list_items(f"{endpoint}/comments"),
        threads=_list_threads(args.repo, args.pr_number),
        review_id=args.review_id,
    )
    json.dump(describe_pr(_api(endpoint)) | {"timeline": timeline}, sys.stdout)


if __name__ == "__main__":
    main()
