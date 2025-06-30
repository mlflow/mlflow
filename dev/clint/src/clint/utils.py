from __future__ import annotations

import ast
import re
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
