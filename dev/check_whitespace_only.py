"""
Detect diff hunks where all changes are whitespace-only.

This helps avoid unnecessary commit history noise from whitespace-only changes.
"""

import argparse
import json
import os
import sys
import urllib.request

BYPASS_LABEL = "allow-whitespace-only"


def github_api_request(url: str, accept: str) -> str:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if github_token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {github_token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def get_pr_diff(owner: str, repo: str, pull_number: int) -> str:
    url = f"https://github.com/{owner}/{repo}/pull/{pull_number}.diff"
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def get_pr_labels(owner: str, repo: str, pull_number: int) -> list[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    data = json.loads(github_api_request(url, "application/vnd.github.v3+json"))
    return [label_obj["name"] for label_obj in data.get("labels", [])]


def parse_diff(diff_text: str | None) -> list[tuple[str, int]]:
    """Parse diff and return list of (file_path, hunk_start_line) for whitespace-only hunks.

    A hunk is considered whitespace-only if all additions and deletions differ only in whitespace.
    """
    if not diff_text:
        return []

    whitespace_hunks: list[tuple[str, int]] = []
    current_file: str | None = None
    hunk_start_line: int | None = None
    added_lines: list[str] = []
    removed_lines: list[str] = []

    def is_hunk_whitespace_only() -> bool:
        """Check if current hunk has only whitespace changes."""
        if not added_lines and not removed_lines:
            return False

        # For each line, compare stripped versions
        # If all changes are just whitespace, the stripped versions should match
        all_lines = added_lines + removed_lines
        return all(line.strip() == "" for line in all_lines) or (
            len(added_lines) == len(removed_lines)
            and all(a.strip() == r.strip() for a, r in zip(added_lines, removed_lines))
        )

    def finalize_hunk():
        """Check and record current hunk if it's whitespace-only."""
        if current_file and hunk_start_line is not None and is_hunk_whitespace_only():
            whitespace_hunks.append((current_file, hunk_start_line))

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            finalize_hunk()
            current_file = None
            hunk_start_line = None
            added_lines = []
            removed_lines = []

        elif line.startswith("--- a/"):
            current_file = None if line == "--- /dev/null" else line[6:]

        elif line.startswith("+++ b/"):
            current_file = None if line == "+++ /dev/null" else line[6:]

        elif line.startswith("@@"):
            # New hunk - finalize previous one first
            finalize_hunk()
            added_lines = []
            removed_lines = []

            # Extract the starting line number from the hunk header
            # Format: @@ -old_start,old_count +new_start,new_count @@
            parts = line.split()
            if len(parts) >= 3:
                new_range = parts[2]  # e.g., "+1,3" or "+1"
                hunk_start_line = int(new_range.split(",")[0].lstrip("+"))
            else:
                hunk_start_line = 1

        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

        elif line.startswith("-") and not line.startswith("---"):
            removed_lines.append(line[1:])

    # Don't forget the last hunk
    finalize_hunk()

    return whitespace_hunks


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
    if hunks := parse_diff(diff_text):
        pr_labels = get_pr_labels(owner, repo, pull_number)
        has_bypass_label = BYPASS_LABEL in pr_labels

        level = "warning" if has_bypass_label else "error"
        message = (
            f"This hunk only has whitespace changes (bypassed with '{BYPASS_LABEL}' label)."
            if has_bypass_label
            else (
                f"This hunk only has whitespace changes. "
                f"Please revert them or apply the '{BYPASS_LABEL}' label to bypass this check "
                f"if they are necessary."
            )
        )

        for file_path, line_number in hunks:
            # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions
            print(f"::{level} file={file_path},line={line_number},col=1::{message}")

        if not has_bypass_label:
            sys.exit(1)


if __name__ == "__main__":
    main()
