from __future__ import annotations

import ast
import functools
import re
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=1)
def get_repo_root() -> Path:
    """Get git repository root path, cached for performance."""
    try:
        return Path(
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("Not in a git repository") from e


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
    Return git-tracked and untracked (but not ignored) files matching the given pathspecs.
    Git does not filter by extension; filtering happens in Python.
    """
    repo_root = get_repo_root()

    # Convert pathspecs to be relative to repository root
    converted_pathspecs = []
    for pathspec in pathspecs:
        if pathspec.is_absolute():
            if pathspec.is_relative_to(repo_root):
                rel_path = pathspec.relative_to(repo_root)
                converted_pathspecs.append(str(rel_path))
            # Skip paths outside repository
        else:
            # Convert relative path from current working directory to relative to repo root
            abs_path = Path.cwd() / pathspec
            if abs_path.is_relative_to(repo_root):
                rel_path = abs_path.relative_to(repo_root)
                converted_pathspecs.append(str(rel_path))
            # Skip paths outside repository

    if not converted_pathspecs:
        return []

    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *converted_pathspecs,
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("Failed to list git files") from e

    return [Path(line) for line in out.splitlines() if line]


def resolve_paths(paths: list[Path]) -> list[Path]:
    """
    Resolve CLI arguments into a list of tracked and untracked files to lint.

    - Includes git-tracked files and untracked files (but not ignored files)
    - Only includes: .md, .mdx, .rst, .py, .ipynb
    """
    if not paths:
        paths = [Path(".")]

    all_files = _git_ls_files(paths)

    filtered = {p for p in all_files if p.suffix.lower() in ALLOWED_EXTS and p.exists()}

    return sorted(filtered)
