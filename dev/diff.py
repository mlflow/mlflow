"""
Fetch and display Python file diffs from a GitHub Pull Request.

Usage:
python dev/py_diff.py --pr https://github.com/mlflow/mlflow/pull/16870
"""

import argparse
import os
import re
from urllib.request import Request, urlopen


def parse_pr_url(pr_url: str) -> tuple[str, str, str]:
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}")

    return match.group(1), match.group(2), match.group(3)


def fetch_pr_diff(owner: str, repo: str, pr_number: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {token}"

    req = Request(url, headers=headers)
    with urlopen(req) as response:
        return response.read().decode("utf-8")


def filter_python_diff(full_diff: str) -> str:
    lines = full_diff.split("\n")
    python_diff: list[str] = []
    in_python_file = False
    for line in lines:
        if line.startswith("diff --git"):
            # Extract file path from: diff --git a/path/to/file.py b/path/to/file.py
            if match := re.match(r"diff --git a/(.*?) b/(.*?)$", line):
                file_path = match.group(2)  # Use the 'b/' path (new file path)
                in_python_file = file_path.endswith(".py")
            else:
                in_python_file = False

            if in_python_file:
                python_diff.append(line)
        elif in_python_file:
            python_diff.append(line)

    # Add line numbers to the diff
    result_lines: list[str] = []
    old_line = 0
    new_line = 0

    for line in python_diff:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Python file diffs from a GitHub Pull Request"
    )
    parser.add_argument(
        "--pr",
        required=True,
        help="GitHub PR URL (e.g., https://github.com/mlflow/mlflow/pull/123)",
    )
    args = parser.parse_args()

    owner, repo, pr_number = parse_pr_url(args.pr)
    full_diff = fetch_pr_diff(owner, repo, pr_number)
    diff = filter_python_diff(full_diff)
    print(diff)


if __name__ == "__main__":
    main()
