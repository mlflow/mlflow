from __future__ import annotations

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
    def check(node: ast.FunctionDef | ast.AsyncFunctionDef, resolver: Resolver) -> bool:
        """
        Returns True if the function has @pytest.mark.repeat decorator.
        """
        return any(
            (res := resolver.resolve(deco)) and res == ["pytest", "mark", "repeat"]
            for deco in node.decorator_list
        )
