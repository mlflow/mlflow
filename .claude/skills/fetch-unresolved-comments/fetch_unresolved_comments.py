"""Fetch unresolved PR review comments using GitHub GraphQL API.

Example usage:
    uv run .claude/skills/fetch-unresolved-comments/fetch_unresolved_comments.py https://github.com/mlflow/mlflow/pull/18327
"""
# ruff: noqa: T201

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen


def get_github_token() -> str:
    """Get GitHub token from environment or gh CLI."""
    if token := os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: GITHUB_TOKEN not found (set env var or install gh CLI)", file=sys.stderr)
        sys.exit(1)


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse GitHub PR URL into owner, repo, and PR number."""
    if m := re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url):
        return m.group(1), m.group(2), int(m.group(3))
    raise ValueError(f"Invalid PR URL: {url}")


def fetch_unresolved_comments(owner: str, repo: str, pr_number: int, token: str) -> dict[str, Any]:
    """Fetch unresolved review threads from a PR using GraphQL."""

    query = """
    query($owner: String!, $repo: String!, $prNumber: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $prNumber) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              isOutdated
              comments(first: 100) {
                nodes {
                  id
                  databaseId
                  body
                  path
                  line
                  startLine
                  diffHunk
                  author {
                    login
                  }
                  createdAt
                  updatedAt
                }
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "owner": owner,
        "repo": repo,
        "prNumber": pr_number,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    data = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request("https://api.github.com/graphql", data=data, headers=headers)

    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        sys.exit(1)


def format_comments(data: dict[str, Any]) -> dict[str, Any]:
    """Format unresolved comments for easier consumption."""

    try:
        threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    except (KeyError, TypeError):
        print("Error: Invalid response structure", file=sys.stderr)
        print(json.dumps(data, indent=2), file=sys.stderr)
        sys.exit(1)

    by_file = {}
    total_comments = 0

    for thread in threads:
        if not thread["isResolved"]:
            comments = []
            path = None
            line = None
            start_line = None
            diff_hunk = None

            for comment in thread["comments"]["nodes"]:
                if path is None:
                    path = comment["path"]
                    line = comment["line"]
                    start_line = comment.get("startLine")
                    diff_hunk = comment.get("diffHunk")

                comments.append(
                    {
                        "id": comment["databaseId"],
                        "body": comment["body"],
                        "author": comment["author"]["login"] if comment["author"] else "unknown",
                        "createdAt": comment["createdAt"],
                    }
                )
                total_comments += 1

            if path:
                if path not in by_file:
                    by_file[path] = []

                by_file[path].append(
                    {
                        "thread_id": thread["id"],
                        "isOutdated": thread["isOutdated"],
                        "line": line,
                        "startLine": start_line,
                        "diffHunk": diff_hunk,
                        "comments": comments,
                    }
                )

    return {
        "total": total_comments,
        "by_file": by_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch unresolved PR review comments using GitHub GraphQL API"
    )
    parser.add_argument(
        "pr_url", help="GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)"
    )

    args = parser.parse_args()

    token = get_github_token()
    owner, repo, pr_number = parse_pr_url(args.pr_url)

    data = fetch_unresolved_comments(owner, repo, pr_number, token)
    formatted = format_comments(data)
    formatted["by_file"] = {
        path: [thread for thread in threads if not thread["isOutdated"]]
        for path, threads in formatted["by_file"].items()
    }
    formatted["by_file"] = {k: v for k, v in formatted["by_file"].items() if v}
    formatted["total"] = sum(
        len(thread["comments"]) for threads in formatted["by_file"].values() for thread in threads
    )

    print(json.dumps(formatted, indent=2))


if __name__ == "__main__":
    main()
