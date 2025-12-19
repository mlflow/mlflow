import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


def _is_deprecation_warning(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Name) and expr.id == "DeprecationWarning"


class ForbiddenDeprecationWarning(Rule):
    def _message(self) -> str:
        return (
            "Do not use `DeprecationWarning` with `warnings.warn()`. "
            "Use `FutureWarning` instead since Python does not show `DeprecationWarning` "
            "by default."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> ast.expr | None:
        """
        Checks if the given node is a call to `warnings.warn` with `DeprecationWarning`.
        """
        # Check if this is a call to `warnings.warn`
        if (resolved := resolver.resolve(node.func)) and resolved == ["warnings", "warn"]:
            # Check if there's a `category` positional argument with `DeprecationWarning`
            if len(node.args) >= 2 and _is_deprecation_warning(node.args[1]):
                return node.args[1]
            # Check if there's a `category` keyword argument with `DeprecationWarning`
            elif kw := next((kw.value for kw in node.keywords if kw.arg == "category"), None):
                if _is_deprecation_warning(kw):
                    return kw

        return None
