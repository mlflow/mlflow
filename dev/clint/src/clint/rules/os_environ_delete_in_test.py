import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class OsEnvironDeleteInTest(Rule):
    def _message(self) -> str:
        return "Do not delete `os.environ` in test directly. Use `monkeypatch.delenv` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.delenv)."

    @staticmethod
    def check(node: ast.Delete, resolver: Resolver) -> bool:
        """
        Returns True if the deletion is from os.environ[...].
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            resolved = resolver.resolve(node.targets[0].value)
            return resolved == ["os", "environ"]
        return False
