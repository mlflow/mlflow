from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


def resolve_expr(expr: ast.expr) -> list[str] | None:
    """
    Resolves `expr` to a list of attribute names. For example, given `expr` like
    `some.module.attribute`, ['some', 'module', 'attribute'] is returned.
    If `expr` is not resolvable, `None` is returned.
    """
    if isinstance(expr, ast.Attribute):
        base = resolve_expr(expr.value)
        if base is None:
            return None
        return base + [expr.attr]
    elif isinstance(expr, ast.Name):
        return [expr.id]

    return None


def get_ignored_rules_for_file(
    file_path: Path, per_file_ignores: dict[re.Pattern[str], set[str]]
) -> set[str]:
    """
    Returns a set of rule names that should be ignored for the given file path.

    Args:
        file_path: The file path to check
        per_file_ignores: Dict mapping compiled regex patterns to lists of rule names to ignore

    Returns:
        Set of rule names to ignore for this file
    """
    ignored_rules: set[str] = set()
    for pattern, rules in per_file_ignores.items():
        if pattern.fullmatch(file_path.as_posix()):
            ignored_rules |= rules
    return ignored_rules


ALLOWED_EXTS = {".md", ".mdx", ".rst", ".py", ".ipynb"}


def _git_ls_files(pathspecs: list[Path]) -> list[Path]:
    """
    Return git-tracked files matching the given pathspecs.
    Git does not filter by extension; filtering happens in Python.
    """
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--", *pathspecs],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("Failed to list git-tracked files") from e

    return [Path(line) for line in out.splitlines() if line]


def _git_ls_untracked_files(pathspecs: list[Path]) -> list[Path]:
    """
    Return git-untracked (but not ignored) files matching the given pathspecs.
    Git does not filter by extension; filtering happens in Python.
    """
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *pathspecs],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("Failed to list git-untracked files") from e

    return [Path(line) for line in out.splitlines() if line]


def resolve_paths(paths: list[Path]) -> list[Path]:
    """
    Resolve CLI arguments into a list of tracked and untracked files to lint.

    - Includes git-tracked files and untracked files (but not ignored files)
    - Only includes: .md, .mdx, .rst, .py, .ipynb
    """
    if not paths:
        paths = [Path(".")]

    tracked = _git_ls_files(paths)
    untracked = _git_ls_untracked_files(paths)

    # Combine tracked and untracked files, removing duplicates
    all_files = tracked + untracked

    filtered = {p for p in all_files if p.suffix.lower() in ALLOWED_EXTS and p.exists()}

    return sorted(filtered)
