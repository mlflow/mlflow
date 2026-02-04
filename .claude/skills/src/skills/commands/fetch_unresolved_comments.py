# ruff: noqa: T201
"""Fetch unresolved PR review comments using GitHub GraphQL API."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from pydantic import BaseModel

from skills.github import GitHubClient, parse_pr_url
from skills.github.types import ReviewComment, ReviewThread


class UnresolvedCommentsResult(BaseModel):
    total: int
    by_file: dict[str, list[ReviewThread]]


REVIEW_THREADS_QUERY = """
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


def format_comments(data: dict[str, Any]) -> UnresolvedCommentsResult:
    """Format unresolved comments grouped by file."""
    threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]

    by_file: dict[str, list[ReviewThread]] = {}
    total_comments = 0

    for thread in threads:
        if thread["isResolved"] or thread["isOutdated"]:
            continue

        comments: list[ReviewComment] = []
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
                ReviewComment(
                    id=comment["databaseId"],
                    body=comment["body"],
                    author=comment["author"]["login"] if comment["author"] else "unknown",
                    createdAt=comment["createdAt"],
                )
            )
            total_comments += 1

        if path and comments:
            if path not in by_file:
                by_file[path] = []
            by_file[path].append(
                ReviewThread(
                    thread_id=thread["id"],
                    line=line,
                    startLine=start_line,
                    diffHunk=diff_hunk,
                    comments=comments,
                )
            )

    return UnresolvedCommentsResult(total=total_comments, by_file=by_file)


async def fetch_unresolved_comments(pr_url: str) -> UnresolvedCommentsResult:
    owner, repo, pr_number = parse_pr_url(pr_url)

    async with GitHubClient() as client:
        data = await client.graphql(
            REVIEW_THREADS_QUERY,
            {"owner": owner, "repo": repo, "prNumber": pr_number},
        )

    return format_comments(data)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "fetch-unresolved-comments",
        help="Fetch unresolved PR review comments",
    )
    parser.add_argument("pr_url", help="GitHub PR URL")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    result = asyncio.run(fetch_unresolved_comments(args.pr_url))
    print(result.model_dump_json(indent=2))
