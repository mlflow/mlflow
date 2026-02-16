import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class PytestMarkRepeat(Rule):
    def _message(self) -> str:
        return (
            "@pytest.mark.repeat decorator should not be committed. "
            "This decorator is meant for local testing only to check for flaky tests."
        )

    @staticmethod
    def check(decorator_list: list[ast.expr], resolver: Resolver) -> ast.expr | None:
        """
        Returns the decorator node if it is a `@pytest.mark.repeat` decorator.
        """
        for deco in decorator_list:
            if (res := resolver.resolve(deco)) and res == ["pytest", "mark", "repeat"]:
                return deco
        return None
