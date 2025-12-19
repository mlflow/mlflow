import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class OsEnvironSetInTest(Rule):
    def _message(self) -> str:
        return "Do not set `os.environ` in test directly. Use `monkeypatch.setenv` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.setenv)."

    @staticmethod
    def check(node: ast.Assign, resolver: Resolver) -> bool:
        """
        Returns True if the assignment is to os.environ[...].
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            resolved = resolver.resolve(node.targets[0].value)
            return resolved == ["os", "environ"]
        return False
