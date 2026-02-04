"""
Lightweight hook for validating code written by Claude Code.
"""

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

KILL_SWITCH_ENV_VAR = "CLAUDE_LINT_HOOK_DISABLED"


@dataclass
class LintError:
    file: Path
    line: int
    column: int
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.message}"


@dataclass
class DiffRange:
    start: int
    end: int

    def overlaps(self, start: int, end: int) -> bool:
        return start <= self.end and self.start <= end


def parse_diff_ranges(diff_output: str) -> list[DiffRange]:
    """Parse unified diff output and extract added line ranges."""
    ranges: list[DiffRange] = []
    for line in diff_output.splitlines():
        if line.startswith("@@ "):
            if match := re.search(r"\+(\d+)(?:,(\d+))?", line):
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                ranges.append(DiffRange(start=start, end=start + count))
    return ranges


def overlaps_with_diff(node: ast.AST, ranges: list[DiffRange]) -> bool:
    return any(r.overlaps(node.lineno, node.end_lineno or node.lineno) for r in ranges)


class Visitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, diff_ranges: list[DiffRange]) -> None:
        self.file_path = file_path
        self.diff_ranges = diff_ranges
        self.errors: list[LintError] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.generic_visit(node)


def lint(file_path: Path, source: str, diff_ranges: list[DiffRange]) -> list[LintError]:
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return [LintError(file=file_path, line=0, column=0, message=f"Failed to parse: {e}")]

    visitor = Visitor(file_path=file_path, diff_ranges=diff_ranges)
    visitor.visit(tree)
    return visitor.errors


def is_test_file(path: Path) -> bool:
    return path.parts[0] == "tests" and path.name.startswith("test_")


@dataclass
class HookInput:
    tool_name: Literal["Edit", "Write"]
    file_path: Path

    @classmethod
    def parse(cls) -> "HookInput | None":
        # https://code.claude.com/docs/en/hooks#posttooluse-input
        data = json.loads(sys.stdin.read())
        tool_name = data.get("tool_name")
        tool_input = data.get("tool_input")
        if tool_name not in ("Edit", "Write"):
            return None
        file_path_str = tool_input.get("file_path")
        if not file_path_str:
            return None
        file_path = Path(file_path_str)
        if project_dir := os.environ.get("CLAUDE_PROJECT_DIR"):
            file_path = file_path.relative_to(project_dir)
        return cls(
            tool_name=tool_name,
            file_path=file_path,
        )


def is_tracked(file_path: Path) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", file_path], capture_output=True)
    return result.returncode == 0


def get_source_and_diff_ranges(hook_input: HookInput) -> tuple[str, list[DiffRange]]:
    if hook_input.tool_name == "Edit" and is_tracked(hook_input.file_path):
        # For Edit on tracked files, use git diff to get only changed lines
        diff_output = subprocess.check_output(
            ["git", "--no-pager", "diff", "-U0", "HEAD", "--", hook_input.file_path],
            text=True,
        )
        diff_ranges = parse_diff_ranges(diff_output)
    else:
        # For Write or Edit on untracked files, lint the whole file
        diff_ranges = [DiffRange(start=1, end=sys.maxsize)]
    source = hook_input.file_path.read_text()
    return source, diff_ranges


def main() -> int:
    # Kill switch: disable hook if environment variable is set
    if os.environ.get(KILL_SWITCH_ENV_VAR):
        return 0

    hook_input = HookInput.parse()
    if not hook_input:
        return 0

    # Ignore non-Python files
    if hook_input.file_path.suffix != ".py":
        return 0

    # Ignore non-test files
    if not is_test_file(hook_input.file_path):
        return 0

    source, diff_ranges = get_source_and_diff_ranges(hook_input)
    if errors := lint(hook_input.file_path, source, diff_ranges):
        error_details = "\n".join(f"  - {error}" for error in errors)
        reason = (
            f"Lint errors found:\n{error_details}\n\n"
            f"To disable this hook, set {KILL_SWITCH_ENV_VAR}=1"
        )
        # Exit code 2 = blocking error. stderr is fed back to Claude.
        # See: https://code.claude.com/docs/en/hooks#hook-output
        sys.stderr.write(reason + "\n")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
