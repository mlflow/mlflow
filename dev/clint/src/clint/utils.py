from __future__ import annotations

import ast


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
