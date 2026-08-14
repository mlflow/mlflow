import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class RedundantMockReturnValue(Rule):
    def _message(self) -> str:
        return (
            "Do not pass `return_value=MagicMock()` or `return_value=Mock()` to `patch()`. "
            "The default return value of a mock is already a new mock."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is a patch() with a redundant return_value=Mock() or
        return_value=MagicMock() (no arguments) keyword argument.
        """
        match resolver.resolve(node.func):
            case ["unittest", "mock", "patch", *_]:
                pass
            case _:
                return False

        for keyword in node.keywords:
            match keyword:
                case ast.keyword(
                    arg="return_value",
                    value=ast.Call(args=[], keywords=[]) as value,
                ):
                    match resolver.resolve(value.func):
                        case ["unittest", "mock", "Mock" | "MagicMock"]:
                            return True

        return False
