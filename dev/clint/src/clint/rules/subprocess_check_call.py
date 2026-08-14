import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class SubprocessCheckCall(Rule):
    def _message(self) -> str:
        return (
            "Use `subprocess.check_call(...)` instead of `subprocess.run(..., check=True)` "
            "for better readability. Only applies when check=True is the only keyword argument."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if `node` is `subprocess.run(..., check=True)` with no other keyword arguments.
        """
        resolved = resolver.resolve(node)

        # Check if this is subprocess.run
        if resolved != ["subprocess", "run"]:
            return False

        # Check if there are any keyword arguments
        if not node.keywords:
            return False

        # Check if the only keyword argument is check=True
        if len(node.keywords) != 1:
            return False

        keyword = node.keywords[0]

        # Check if the keyword is 'check' (not **kwargs)
        if keyword.arg != "check":
            return False

        # Check if the value is True
        if not isinstance(keyword.value, ast.Constant):
            return False

        return keyword.value.value is True
