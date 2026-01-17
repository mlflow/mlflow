"""
Detect files where all changes are whitespace-only.

This helps avoid unnecessary commit history noise from whitespace-only changes.
"""

import argparse
import json
import os
import sys
import urllib.request
from typing import cast

BYPASS_LABEL = "allow-whitespace-only"


def github_api_request(url: str, accept: str) -> str:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if github_token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {github_token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return cast(str, response.read().decode("utf-8"))


def get_pr_diff(owner: str, repo: str, pull_number: int) -> str:
    url = f"https://github.com/{owner}/{repo}/pull/{pull_number}.diff"
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=30) as response:
        return cast(str, response.read().decode("utf-8"))


def get_pr_labels(owner: str, repo: str, pull_number: int) -> list[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    data = json.loads(github_api_request(url, "application/vnd.github.v3+json"))
    return [label_obj["name"] for label_obj in data.get("labels", [])]


def parse_diff(diff_text: str | None) -> list[str]:
    if not diff_text:
        return []

    files: list[str] = []
    current_file: str | None = None
    changes: list[str] = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_file and changes and all(c.strip() == "" for c in changes):
                files.append(current_file)

            current_file = None
            changes = []

        elif line.startswith("--- a/"):
            current_file = None if line == "--- /dev/null" else line[6:]

        elif line.startswith("+++ b/"):
            current_file = None if line == "+++ /dev/null" else line[6:]

        elif line.startswith("+") or line.startswith("-"):
            content = line[1:]
            changes.append(content)

    if current_file and changes and all(c.strip() == "" for c in changes):
        files.append(current_file)

    return files


def parse_args() -> tuple[str, str, int]:
    parser = argparse.ArgumentParser(
        description="Check for unnecessary whitespace-only changes in the diff"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help='Repository in the format "owner/repo" (e.g., "mlflow/mlflow")',
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        help="Pull request number",
    )
    args = parser.parse_args()

    owner, repo = args.repo.split("/")
    return owner, repo, args.pr


def main() -> None:
    owner, repo, pull_number = parse_args()
    diff_text = get_pr_diff(owner, repo, pull_number)
    if files := parse_diff(diff_text):
        pr_labels = get_pr_labels(owner, repo, pull_number)
        has_bypass_label = BYPASS_LABEL in pr_labels

        level = "warning" if has_bypass_label else "error"
        message = (
            f"This file only has whitespace changes (bypassed with '{BYPASS_LABEL}' label)."
            if has_bypass_label
            else (
                f"This file only has whitespace changes. "
                f"Please revert them or apply the '{BYPASS_LABEL}' label to bypass this check "
                f"if they are necessary."
            )
        )

        for file_path in files:
            # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions
            print(f"::{level} file={file_path},line=1,col=1::{message}")

        if not has_bypass_label:
            sys.exit(1)


if __name__ == "__main__":
    main()
