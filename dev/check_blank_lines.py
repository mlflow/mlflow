"""
Detect files where all changes are blank lines.

This helps avoid unnecessary commit history noise from blank line-only changes.
"""

import argparse
import os
import urllib.request


def get_diff_from_github_api(owner: str, repo: str, pull_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Add authorization if token is available
    if github_token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {github_token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_diff(diff_text: str | None) -> list[str]:
    if not diff_text:
        return []

    files: list[str] = []
    current_file: str | None = None
    changes: list[str] = []

    for line in diff_text.split("\n"):
        # New file
        if line.startswith("diff --git"):
            # Process previous file - flag if all changes are blank
            if current_file and changes and all(c.strip() == "" for c in changes):
                files.append(current_file)

            # Reset for new file
            current_file = None
            changes = []

        elif line.startswith("--- a/"):
            # Extract file path (skip new files indicated by /dev/null)
            current_file = None if line == "--- /dev/null" else line[6:]

        elif line.startswith("+++ b/"):
            # This is the new file path, use this one (skip deleted files)
            current_file = None if line == "+++ /dev/null" else line[6:]

        elif line.startswith("+") or line.startswith("-"):
            # Collect the content of additions and deletions
            content = line[1:]  # Remove +/- prefix
            changes.append(content)

    # Process last file
    if current_file and changes and all(c.strip() == "" for c in changes):
        files.append(current_file)

    return files


def parse_args() -> tuple[str, str, int]:
    parser = argparse.ArgumentParser(
        description="Check for unnecessary blank line changes in the diff"
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
    diff_text = get_diff_from_github_api(owner, repo, pull_number)
    if files := parse_diff(diff_text):
        message = (
            "This file only has blank line changes. "
            "If unintentional, please revert them to avoid unnecessary commit history noise."
        )
        for file_path in files:
            # https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#setting-a-warning-message
            print(f"::warning file={file_path},line=1,col=1::{message}")


if __name__ == "__main__":
    main()
