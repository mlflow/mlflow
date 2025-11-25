"""
Lightweight hook for validating code written by Claude Code.
"""

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

DefNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass
class LintError:
    file: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.message}"


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
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                ranges.append(DiffRange(start=start, end=start + count))
    return ranges


def overlaps_with_diff(node: ast.Constant, ranges: list[DiffRange]) -> bool:
    return any(r.overlaps(node.lineno, node.end_lineno or node.lineno) for r in ranges)


def get_docstring_node(node: DefNode) -> ast.Constant | None:
    match node.body:
        case [ast.Expr(value=ast.Constant(value=str()) as const), *_]:
            return const
    return None


def is_redundant_docstring(node: DefNode) -> bool:
    docstring = ast.get_docstring(node)
    if not docstring:
        return False
    return "\n" not in docstring.strip()


class Visitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, diff_ranges: list[DiffRange]) -> None:
        self.file_path = file_path
        self.diff_ranges = diff_ranges
        self.errors: list[LintError] = []

    def _check_docstring(self, node: DefNode) -> None:
        docstring_node = get_docstring_node(node)
        if not docstring_node:
            return
        if not overlaps_with_diff(docstring_node, self.diff_ranges):
            return
        if is_redundant_docstring(node):
            self.errors.append(
                LintError(
                    file=self.file_path,
                    line=docstring_node.lineno,
                    message=f"Redundant docstring in '{node.name}'",
                )
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_docstring(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_docstring(node)
        self.generic_visit(node)


def lint(file_path: Path, source: str, diff_ranges: list[DiffRange]) -> list[LintError]:
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return [LintError(file=file_path, line=0, message=f"Failed to parse: {e}")]

    visitor = Visitor(file_path=file_path, diff_ranges=diff_ranges)
    visitor.visit(tree)
    return visitor.errors


def is_test_file(path: Path) -> bool:
    return path.parts[0] == "tests" and path.name.startswith("test_")


ToolName: TypeAlias = Literal["Edit", "Write"]


@dataclass
class HookInput:
    tool_name: ToolName
    file_path: Path
    content: str | None

    @classmethod
    def parse(cls) -> "HookInput | None":
        data = json.loads(sys.stdin.read())
        tool_name = data.get("tool_name")
        tool_input = data.get("tool_input")
        if tool_name not in ("Edit", "Write") or not tool_input:
            return None
        file_path_str = tool_input.get("file_path")
        if not file_path_str:
            return None
        return cls(
            tool_name=tool_name,
            file_path=Path(file_path_str),
            content=tool_input.get("content"),
        )


def get_source_and_diff_ranges(hook_input: HookInput) -> tuple[str, list[DiffRange]] | None:
    if hook_input.tool_name == "Edit":
        # For Edit, use git diff to get only changed lines
        diff_output = subprocess.check_output(
            ["git", "--no-pager", "diff", "-U0", "HEAD", "--", hook_input.file_path],
            text=True,
        )
        diff_ranges = parse_diff_ranges(diff_output)
        if not diff_ranges:
            return None
        source = hook_input.file_path.read_text()
    else:
        # For Write, lint all lines using the provided content
        source = hook_input.content or ""
        diff_ranges = [DiffRange(start=1, end=999999)]
    return source, diff_ranges


def main() -> int:
    hook_input = HookInput.parse()
    if not hook_input:
        return 0

    # Only lint test files
    if hook_input.file_path.suffix != ".py" or not is_test_file(hook_input.file_path):
        return 0

    result = get_source_and_diff_ranges(hook_input)
    if not result:
        return 0

    source, diff_ranges = result
    errors = lint(hook_input.file_path, source, diff_ranges)

    if errors:
        error_details = "\n".join(f"  - {error}" for error in errors)
        reason = f"Lint errors found:\n{error_details}"
        # Exit code 2 = blocking error. stderr is fed back to Claude.
        # See: https://code.claude.com/docs/en/hooks#hook-output
        sys.stderr.write(reason + "\n")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
