"""
Usage:

```
python outputs/diff.py --pr https://github.com/mlflow/mlflow/pull/17230 > diff.txt
```

Output format:

The script generates a side-by-side diff view with line numbers and change markers:

```
File: path/to/file.py
=============================================================================
                    Old                 │             New
────────────────────────────────────────┼────────────────────────────────────
   10  - old_line_content               │   10  + new_line_content
   11    unchanged_line                 │   11    unchanged_line
   12  - deleted_line                   │
                                        │   12  + added_line
```
"""

import argparse
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class DiffLine:
    old_line_num: int | None
    new_line_num: int | None
    change_type: Literal["add", "delete", "context"]
    content: str


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine]


@dataclass
class FileDiff:
    old_file: Path
    new_file: Path
    hunks: list[DiffHunk]


class DiffParser:
    def __init__(self, diff_text: str) -> None:
        self.lines = diff_text.split("\n")
        self.current_line = 0

    @staticmethod
    def parse_diff_hunk(hunk_text: str) -> DiffHunk | None:
        parser = DiffParser(hunk_text)
        return parser.parse_hunk()

    def parse(self) -> list[FileDiff]:
        file_diffs: list[FileDiff] = []

        while self.current_line < len(self.lines):
            if self.lines[self.current_line].startswith("diff --git"):
                file_diff = self._parse_file_diff()
                if file_diff:
                    file_diffs.append(file_diff)
            else:
                self.current_line += 1

        return file_diffs

    def _parse_file_diff(self) -> FileDiff | None:
        diff_line = self.lines[self.current_line]
        match = re.match(r"^diff --git a/(.*) b/(.*)$", diff_line)
        if not match:
            self.current_line += 1
            return None

        old_file = Path(match.group(1))
        new_file = Path(match.group(2))
        self.current_line += 1

        # Check for binary files
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            if line.startswith("Binary files"):
                # Skip binary file
                self.current_line += 1
                # Continue to next file
                while self.current_line < len(self.lines):
                    if self.lines[self.current_line].startswith("diff --git"):
                        break
                    self.current_line += 1
                return None  # Skip binary files
            if line.startswith("---") or line.startswith("+++"):
                break
            if line.startswith("@@"):
                break
            if line.startswith("diff --git"):
                return FileDiff(old_file=old_file, new_file=new_file, hunks=[])
            self.current_line += 1

        # Skip --- and +++ lines
        if self.current_line < len(self.lines) and self.lines[self.current_line].startswith("---"):
            self.current_line += 1
        if self.current_line < len(self.lines) and self.lines[self.current_line].startswith("+++"):
            self.current_line += 1

        # Parse hunks
        hunks: list[DiffHunk] = []
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            if line.startswith("@@"):
                hunk = self.parse_hunk()
                if hunk:
                    hunks.append(hunk)
            elif line.startswith("diff --git"):
                break
            else:
                self.current_line += 1

        return FileDiff(old_file=old_file, new_file=new_file, hunks=hunks)

    def parse_hunk(self) -> DiffHunk | None:
        hunk_header = self.lines[self.current_line]
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", hunk_header)
        if not match:
            self.current_line += 1
            return None

        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1

        self.current_line += 1

        lines: list[DiffLine] = []
        old_line_num = old_start
        new_line_num = new_start

        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]

            if line.startswith("@@") or line.startswith("diff --git"):
                break

            if line.startswith("-"):
                lines.append(
                    DiffLine(
                        old_line_num=old_line_num,
                        new_line_num=None,
                        change_type="delete",
                        content=line[1:] if len(line) > 1 else "",
                    )
                )
                old_line_num += 1
            elif line.startswith("+"):
                lines.append(
                    DiffLine(
                        old_line_num=None,
                        new_line_num=new_line_num,
                        change_type="add",
                        content=line[1:] if len(line) > 1 else "",
                    )
                )
                new_line_num += 1
            elif line.startswith(" "):
                lines.append(
                    DiffLine(
                        old_line_num=old_line_num,
                        new_line_num=new_line_num,
                        change_type="context",
                        content=line[1:] if len(line) > 1 else "",
                    )
                )
                old_line_num += 1
                new_line_num += 1
            elif line.startswith("\\"):
                # "No newline at end of file" - skip
                pass
            else:
                # Context line without space prefix
                lines.append(
                    DiffLine(
                        old_line_num=old_line_num,
                        new_line_num=new_line_num,
                        change_type="context",
                        content=line,
                    )
                )
                old_line_num += 1
                new_line_num += 1

            self.current_line += 1

        return DiffHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=lines,
        )


class TextFormatter:
    def __init__(self, line_width: int = 80, line_num_width: int = 4) -> None:
        self.line_width = line_width
        self.line_num_width = line_num_width
        self.marker_width = 2
        self.content_width = line_width - self.line_num_width - self.marker_width - 1

    @staticmethod
    def calculate_line_num_width(file_diffs: list[FileDiff]) -> int:
        """Calculate the required width for line numbers based on the maximum line number."""
        max_line = 0
        for file_diff in file_diffs:
            for hunk in file_diff.hunks:
                # Check the starting line numbers and counts
                max_line = max(max_line, hunk.old_start + hunk.old_count - 1)
                max_line = max(max_line, hunk.new_start + hunk.new_count - 1)
                # Also check actual line numbers in the diff
                for line in hunk.lines:
                    if line.old_line_num:
                        max_line = max(max_line, line.old_line_num)
                    if line.new_line_num:
                        max_line = max(max_line, line.new_line_num)

        # Return the width needed: minimum 4, or actual width + 1 for padding
        return max(4, len(str(max_line)) + 1)

    def format_file_diff(self, file_diff: FileDiff) -> str:
        lines: list[str] = []

        # Check if this is a new file (created file)
        is_new_file = self._is_new_file(file_diff)

        # File header
        file_name = file_diff.new_file or file_diff.old_file
        lines.append(f"\nFile: {file_name}")

        if is_new_file:
            # For new files, only show the New column
            lines.append("=" * self.line_width)
            lines.append(self._center("New (created)", self.line_width))
            lines.append("─" * self.line_width)

            # Process hunks for new file
            for i, hunk in enumerate(file_diff.hunks):
                if i > 0:
                    lines.append(" " * self.line_width)

                for line in hunk.lines:
                    formatted_lines = self._format_single_line(line, False)
                    lines.extend(formatted_lines)
        else:
            # For modified files, show both columns
            lines.append("=" * (self.line_width * 2 + 3))
            old_header = self._center("Old", self.line_width)
            new_header = self._center("New", self.line_width)
            lines.append(f"{old_header} │ {new_header}")
            lines.append("─" * self.line_width + "─┼─" + "─" * self.line_width)

            # Process hunks
            for i, hunk in enumerate(file_diff.hunks):
                # Add hunk separator if not first
                if i > 0:
                    lines.append(" " * self.line_width + " │ " + " " * self.line_width)

                # Process lines in pairs
                paired_lines = self._pair_diff_lines(hunk.lines)
                for old_line, new_line in paired_lines:
                    formatted_lines = self._format_line_pair(old_line, new_line)
                    lines.extend(formatted_lines)

        return "\n".join(line.rstrip() for line in lines)

    def _is_new_file(self, file_diff: FileDiff) -> bool:
        """Check if this is a newly created file."""
        # A new file typically has all additions and no deletions
        if not file_diff.hunks:
            return False

        for hunk in file_diff.hunks:
            for line in hunk.lines:
                if line.change_type == "delete":
                    return False
                if line.change_type == "context":
                    return False

        # If we only have additions, it's a new file
        return True

    def _pair_diff_lines(
        self, lines: list[DiffLine]
    ) -> list[tuple[DiffLine | None, DiffLine | None]]:
        pairs: list[tuple[DiffLine | None, DiffLine | None]] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.change_type == "context":
                pairs.append((line, line))
                i += 1
            else:
                # Collect consecutive deletes and adds
                delete_lines: list[DiffLine] = []
                add_lines: list[DiffLine] = []

                # Collect all consecutive deletes
                while i < len(lines) and lines[i].change_type == "delete":
                    delete_lines.append(lines[i])
                    i += 1

                # Collect all consecutive adds
                while i < len(lines) and lines[i].change_type == "add":
                    add_lines.append(lines[i])
                    i += 1

                # Pair them up
                max_length = max(len(delete_lines), len(add_lines))
                for j in range(max_length):
                    old_line = delete_lines[j] if j < len(delete_lines) else None
                    new_line = add_lines[j] if j < len(add_lines) else None
                    pairs.append((old_line, new_line))

        return pairs

    def _format_line_pair(self, old_line: DiffLine | None, new_line: DiffLine | None) -> list[str]:
        old_lines = self._format_single_line(old_line, True)
        new_lines = self._format_single_line(new_line, False)

        # Handle different number of wrapped lines by padding with empty lines
        max_lines = max(len(old_lines), len(new_lines))
        empty_old = " " * self.line_width
        empty_new = " " * self.line_width

        result = []
        for i in range(max_lines):
            old_text = old_lines[i] if i < len(old_lines) else empty_old
            new_text = new_lines[i] if i < len(new_lines) else empty_new
            result.append(f"{old_text} │ {new_text}")

        return result

    def _format_single_line(self, line: DiffLine | None, is_old: bool) -> list[str]:
        """Format a single diff line, returning multiple lines if wrapped."""
        if not line:
            return [" " * self.line_width]

        # Line number
        line_num = line.old_line_num if is_old else line.new_line_num
        if line_num is not None:
            line_num_str = str(line_num).rjust(self.line_num_width) + " "
        else:
            line_num_str = " " * (self.line_num_width + 1)

        # Change marker
        marker = "  "
        if line.change_type == "delete" and is_old:
            marker = "- "
        elif line.change_type == "add" and not is_old:
            marker = "+ "

        # Wrap content
        content = line.content or ""
        wrapped_lines = self._wrap_content(content)

        formatted_lines = []
        for i, wrapped_content in enumerate(wrapped_lines):
            if i == 0:
                # First line gets line number and marker
                prefix = line_num_str + marker
            else:
                # Continuation lines get spaces instead of line number/marker
                prefix = " " * (self.line_num_width + 1 + self.marker_width)

            padded_content = wrapped_content.ljust(self.content_width)
            formatted_lines.append(prefix + padded_content)

        return formatted_lines

    def _wrap_content(self, content: str) -> list[str]:
        """Wrap content by words, preserving structure."""
        if len(content) <= self.content_width:
            return [content]

        # Use textwrap to break by words
        wrapped_lines = textwrap.wrap(
            content, width=self.content_width, break_long_words=False, break_on_hyphens=True
        )

        # Handle case where a single word is too long
        if not wrapped_lines and content:
            # Force break very long words
            wrapped_lines = textwrap.wrap(content, width=self.content_width, break_long_words=True)

        return wrapped_lines or [content]

    def _center(self, text: str, width: int) -> str:
        padding = max(0, width - len(text))
        left_pad = padding // 2
        right_pad = padding - left_pad
        return " " * left_pad + text + " " * right_pad

    def format_diff(self, file_diffs: list[FileDiff]) -> str:
        return "\n".join(self.format_file_diff(fd) for fd in file_diffs)


def format_diff(diff_hunk: str) -> str:
    """Format complete diff text into readable output."""
    parser = DiffParser(diff_hunk)
    file_diffs = parser.parse()
    line_num_width = TextFormatter.calculate_line_num_width(file_diffs)
    formatter = TextFormatter(line_num_width=line_num_width)
    return formatter.format_diff(file_diffs)


def get_pr_diff(pr_url: str) -> str:
    # Parse GitHub PR URL to extract owner, repo, and PR number
    # Format: https://github.com/{owner}/{repo}/pull/{number}
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        print(f"Invalid GitHub PR URL format: {pr_url}", file=sys.stderr)
        sys.exit(1)

    owner, repo, pr_number = match.groups()

    # Use GitHub API to get the diff
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"

    # Create request with authentication if GITHUB_TOKEN is available
    request = Request(api_url)
    request.add_header("Accept", "application/vnd.github.v3.diff")
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        request.add_header("Authorization", f"token {github_token}")

    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        print(
            f"Error fetching PR diff from GitHub API: HTTP {e.code} - {e.reason}", file=sys.stderr
        )
        if e.code == 404:
            if not github_token:
                print(
                    "Note: If this is a private repository, set GITHUB_TOKEN environment variable",
                    file=sys.stderr,
                )
            else:
                print(
                    "Note: The PR may not exist or you may not have access to this repository",
                    file=sys.stderr,
                )
        elif e.code == 401:
            print("Note: Authentication failed. Check your GITHUB_TOKEN", file=sys.stderr)
        elif e.code == 403:
            print(
                "Note: Access forbidden. You may have hit the rate limit or lack permissions",
                file=sys.stderr,
            )
        sys.exit(1)
    except URLError as e:
        print(f"Error fetching PR diff from GitHub API: {e}", file=sys.stderr)
        print("Note: Check your network connection and the PR URL", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Format Git diffs into readable side-by-side output"
    )
    parser.add_argument(
        "--pr", type=str, required=True, help="GitHub PR URL to fetch and format diff"
    )
    parser.add_argument(
        "--width", type=int, default=100, help="Line width for formatting (default: 80)"
    )
    parser.add_argument("--file", type=str, help="Only show diff for the specified file")

    args = parser.parse_args()

    # Get diff text from PR URL or stdin
    diff_text = get_pr_diff(args.pr)

    if not diff_text.strip():
        print("No diff data provided.", file=sys.stderr)
        sys.exit(1)

    # Print PR URL if provided
    if args.pr:
        print(f"PR: {args.pr}")
        print()

    print("Note: Long lines are wrapped to fit the specified width.")

    # Parse the diff first
    diff_parser = DiffParser(diff_text)
    file_diffs = diff_parser.parse()

    # Exclude non-python files
    file_diffs = [f for f in file_diffs if f.new_file.suffix == ".py" or f.old_file.suffix == ".py"]

    # Calculate the required line number width
    line_num_width = TextFormatter.calculate_line_num_width(file_diffs)

    # Create formatter with the calculated line number width
    formatter = TextFormatter(line_width=args.width, line_num_width=line_num_width)
    formatted_output = formatter.format_diff(file_diffs)
    print(formatted_output)


if __name__ == "__main__":
    main()
