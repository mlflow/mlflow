import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class ForbiddenSetActiveModelUsage(Rule):
    def _message(self) -> str:
        return (
            "Usage of `set_active_model` is not allowed in mlflow, use `_set_active_model` instead."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """Check if this is a call to set_active_model function."""
        return (
            (resolved := resolver.resolve(node))
            and len(resolved) >= 1
            and resolved[0] == "mlflow"
            and resolved[-1] == "set_active_model"
        )
