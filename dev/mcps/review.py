# /// script
# dependencies = [
#   "fastmcp==2.11.3",
# ]
# ///
"""
GitHub PR Review MCP Server - Tools for automated PR code reviews.

Provides MCP tools for fetching PR diffs with line numbers and adding review comments.
"""

import functools
import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastmcp import FastMCP


def github_api_request(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    accept_header: str = "application/vnd.github.v3+json",
) -> dict[str, Any] | str:
    """Make a request to the GitHub API."""
    headers = {"Accept": accept_header}

    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {token}"

    if data is not None:
        headers["Content-Type"] = "application/json"
        req = Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method=method,
        )
    else:
        req = Request(url, headers=headers, method=method)

    try:
        with urlopen(req) as response:
            content = response.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Error fetching GitHub API: {e.code} {e.reason} {body}") from e

    if accept_header == "application/vnd.github.v3.diff":
        return content
    return json.loads(content)


@functools.lru_cache(maxsize=32)
def fetch_pr_diff(owner: str, repo: str, pull_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    return github_api_request(url, accept_header="application/vnd.github.v3.diff")


def should_exclude_file(file_path: str) -> bool:
    """
    Determine if a file should be excluded from the diff.

    Excludes:
    - .py and .pyi files in mlflow/protos/
    - Auto-generated lock files
      - uv.lock
      - yarn.lock
      - package-lock.json
    - .java files
    - .ipynb files
    """
    path = Path(file_path)

    # Check if it's a Python file in mlflow/protos/
    if path.suffix in {".py", ".pyi"} and path.is_relative_to("mlflow/protos"):
        return True

    # Check for auto-generated lock files
    if path.name in {"uv.lock", "yarn.lock", "package-lock.json"}:
        return True

    # Check for Java files
    if path.suffix == ".java":
        return True

    # Check for Jupyter notebook files
    if path.suffix == ".ipynb":
        return True

    return False


def filter_diff(full_diff: str) -> str:
    lines = full_diff.split("\n")
    filtered_diff: list[str] = []
    in_included_file = False
    for line in lines:
        if line.startswith("diff --git"):
            # Extract file path from: diff --git a/path/to/file.py b/path/to/file.py
            if match := re.match(r"diff --git a/(.*?) b/(.*?)$", line):
                file_path = match.group(2)  # Use the 'b/' path (new file path)
                # Exclude deleted files (where b/ path is dev/null)
                if file_path == "dev/null":
                    in_included_file = False
                else:
                    in_included_file = not should_exclude_file(file_path)
            else:
                in_included_file = False

            if in_included_file:
                filtered_diff.append(line)
        elif in_included_file:
            filtered_diff.append(line)

    # Add line numbers to the diff
    result_lines: list[str] = []
    old_line = 0
    new_line = 0

    for line in filtered_diff:
        if line.startswith("@@"):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
            result_lines.append(line)
        elif line.startswith("diff --git"):
            # Add blank line before new file diff for readability
            if result_lines:  # Don't add blank line before the first diff
                result_lines.append("")
            result_lines.append(line)
        elif line.startswith("index ") or line.startswith("---") or line.startswith("+++"):
            result_lines.append(line)
        elif line.startswith("-"):
            # Deleted line - show old line number on the left (like GitHub)
            result_lines.append(f"{old_line:5d}       | {line}")
            old_line += 1
        elif line.startswith("+"):
            # Added line - show new line number on the right (like GitHub)
            result_lines.append(f"      {new_line:5d} | {line}")
            new_line += 1
        else:
            # Unchanged line - show both line numbers (like GitHub)
            result_lines.append(f"{old_line:5d} {new_line:5d} | {line}")
            old_line += 1
            new_line += 1

    return "\n".join(result_lines)


mcp = FastMCP("Review MCP")


@mcp.tool
def fetch_diff(
    owner: Annotated[str, "Repository owner"],
    repo: Annotated[str, "Repository name"],
    pull_number: Annotated[int, "Pull request number"],
) -> str:
    """
    Fetch the diff of a pull request, excluding certain file types,
    and display it with line numbers.

    Example output:

    ```
    diff --git a/path/to/file.py b/path/to/file.py
    index abc123..def456 100644
    --- a/path/to/file.py
    +++ b/path/to/file.py
    @@ -10,7 +10,7 @@
    10    10 |  import os
    11    11 |  import sys
    12    12 |  from typing import Optional
    13       | -from old_module import OldClass
          14 | +from new_module import NewClass
    14    15 |
    15    16 |  def process_data(input_file: str) -> dict:
    ```
    """
    full_diff = fetch_pr_diff(owner, repo, pull_number)
    return filter_diff(full_diff)


def fetch_pr_head_commit(owner: str, repo: str, pull_number: int) -> str:
    """Fetch the head commit SHA of a pull request."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    pr_data = github_api_request(url)
    return pr_data["head"]["sha"]


@mcp.tool
def add_pr_review_comment(
    owner: Annotated[str, "Repository owner"],
    repo: Annotated[str, "Repository name"],
    pull_number: Annotated[int, "Pull request number"],
    path: Annotated[str, " The relative path to the file that necessitates a comment"],
    body: Annotated[str, "The text of the review comment"],
    line: Annotated[
        int,
        (
            "The line of the blob in the pull request diff that the comment applies to. "
            "For multi-line comments, the last line of the range"
        ),
    ],
    start_line: Annotated[
        int | None,
        "For multi-line comments, the first line of the range that the comment applies to",
    ] = None,
    side: Annotated[
        Literal["LEFT", "RIGHT"],
        "The side of the diff to comment on. 'LEFT' indicates the previous state, 'RIGHT' "
        "indicates the new state",
    ] = "RIGHT",
    start_side: Annotated[
        Literal["LEFT", "RIGHT"] | None,
        (
            "The starting side of the diff to comment on. 'LEFT' indicates the previous state, "
            "'RIGHT' indicates the new state"
        ),
    ] = None,
    subject_type: Annotated[
        Literal["line", "file"],
        (
            "The level at which the comment is targeted. 'line' indicates a specific line, "
            "'file' indicates the entire file"
        ),
    ] = "line",
    in_reply_to: Annotated[
        int | None,
        "The ID of the review comment to reply to. Use this to create a threaded reply",
    ] = None,
) -> str:
    """
    Add a review comment to a pull request.

    https://docs.github.com/en/rest/pulls/comments?apiVersion=2022-11-28#create-a-review-comment-for-a-pull-request
    """
    # First, fetch the head commit SHA
    commit_id = fetch_pr_head_commit(owner, repo, pull_number)
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    data = {
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "body": body,
        "side": side,
    }
    if start_line is not None:
        data["start_line"] = start_line
    if start_side is not None:
        data["start_side"] = start_side
    if subject_type == "file":
        data["subject_type"] = subject_type
    if in_reply_to is not None:
        data["in_reply_to"] = in_reply_to

    result = github_api_request(url, method="POST", data=data)
    return f"Comment added successfully: {result.get('html_url')}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
