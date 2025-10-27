import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class UnnamedThread(Rule):
    def _message(self) -> str:
        return (
            "`threading.Thread()` must be called with a `name` argument to improve debugging "
            "and traceability of thread-related issues."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is threading.Thread() without a name parameter.
        """
        if names := resolver.resolve(node):
            return names == ["threading", "Thread"] and not any(
                keyword.arg == "name" for keyword in node.keywords
            )
        return False
