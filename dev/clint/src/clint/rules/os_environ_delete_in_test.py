import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class OsEnvironDeleteInTest(Rule):
    def _message(self) -> str:
        return (
            "Do not delete `os.environ` in test directly (del os.environ[...] or "
            "os.environ.pop(...)). Use `monkeypatch.delenv` "
            "(https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.delenv)."
        )

    @staticmethod
    def check(node: ast.Delete | ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the operation is deletion from os.environ[...] or
        a call to os.environ.pop().
        """
        if isinstance(node, ast.Delete):
            # Handle: del os.environ["KEY"]
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                resolved = resolver.resolve(node.targets[0].value)
                return resolved == ["os", "environ"]
        elif isinstance(node, ast.Call):
            # Handle: os.environ.pop("KEY")
            resolved = resolver.resolve(node)
            return resolved == ["os", "environ", "pop"]
        return False
