import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class TempDirInTest(Rule):
    def _message(self) -> str:
        return "Do not use `tempfile.TemporaryDirectory` in test directly. Use `tmp_path` fixture (https://docs.pytest.org/en/stable/reference/reference.html#tmp-path)."

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is to tempfile.TemporaryDirectory().
        """
        return resolver.resolve(node) == ["tempfile", "TemporaryDirectory"]
