import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class OsChdirInTest(Rule):
    def _message(self) -> str:
        return "Do not use `os.chdir` in test directly. Use `monkeypatch.chdir` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.chdir)."

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is to os.chdir().
        """
        return resolver.resolve(node) == ["os", "chdir"]
