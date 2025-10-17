import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class MockPatchAsDecorator(Rule):
    def _message(self) -> str:
        return (
            "Do not use `unittest.mock.patch` as a decorator. "
            "Use it as a context manager to avoid patches being active longer than needed "
            "and to make it clear which code depends on them."
        )

    @staticmethod
    def check(decorator_list: list[ast.expr], resolver: Resolver) -> ast.expr | None:
        """
        Returns the decorator node if it is a `@mock.patch` or `@patch` decorator.
        """
        for deco in decorator_list:
            if res := resolver.resolve(deco):
                match res:
                    # Resolver returns ["unittest", "mock", "patch", ...]
                    # The *_ captures variants like "object", "dict", etc.
                    case ["unittest", "mock", "patch", *_]:
                        return deco
        return None
