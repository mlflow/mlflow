"""
Lightweight hook for validating code written by Claude Code.
"""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

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


def parse_diff_ranges(diff_output: str) -> dict[Path, list[DiffRange]]:
    """
    Parse unified diff output and extract added line ranges per file.

    Returns a dict mapping file paths to lists of line ranges that were added.
    """
    file_ranges: dict[Path, list[DiffRange]] = {}
    current_file: Path | None = None

    for line in diff_output.splitlines():
        if line.startswith("+++ b/"):
            current_file = Path(line[6:])
            file_ranges[current_file] = []
        elif line.startswith("@@ ") and current_file is not None:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                file_ranges[current_file].append(DiffRange(start=start, end=start + count))

    return file_ranges


def get_diff_ranges() -> dict[Path, list[DiffRange]]:
    output = subprocess.check_output(
        ["git", "--no-pager", "diff", "-U0", "HEAD", "--", "*.py"], text=True
    )
    return parse_diff_ranges(output)


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


def lint_file(file_path: Path, diff_ranges: list[DiffRange]) -> list[LintError]:
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, OSError) as e:
        return [LintError(file=file_path, line=0, message=f"Failed to parse: {e}")]

    visitor = Visitor(file_path=file_path, diff_ranges=diff_ranges)
    visitor.visit(tree)
    return visitor.errors


def main() -> int:
    try:
        file_diff_ranges = get_diff_ranges()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error running git diff: {e}\n")
        return 1

    if not file_diff_ranges:
        return 0

    all_errors: list[LintError] = []
    for file_path, ranges in file_diff_ranges.items():
        if (
            file_path.suffix == ".py"
            and file_path.exists()
            and file_path.parts[0] == "tests"
            and file_path.name.startswith("test_")
        ):
            errors = lint_file(file_path, ranges)
            all_errors.extend(errors)

    if all_errors:
        error_details = "\n".join(f"  - {error}" for error in all_errors)
        reason = f"Lint errors found:\n{error_details}"
        # Exit code 2 = blocking error. stderr is fed back to Claude.
        # See: https://code.claude.com/docs/en/hooks#hook-output
        sys.stderr.write(reason + "\n")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
