import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class MockPatchDictEnviron(Rule):
    def _message(self) -> str:
        return (
            "Do not use `mock.patch.dict` to modify `os.environ` in tests; "
            "use pytest's monkeypatch fixture (monkeypatch.setenv / monkeypatch.delenv) instead."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is to mock.patch.dict with "os.environ" or os.environ as first arg.
        Handles:
        - mock.patch.dict("os.environ", {...})
        - mock.patch.dict(os.environ, {...})
        - @mock.patch.dict("os.environ", {...})
        """
        if not isinstance(node, ast.Call):
            return False

        # Check if this is mock.patch.dict
        resolved = resolver.resolve(node.func)
        if resolved != ["unittest", "mock", "patch", "dict"]:
            return False

        # Check if the first argument is "os.environ" (string) or os.environ (expression)
        if not node.args:
            return False

        first_arg = node.args[0]

        # Check for string literal "os.environ"
        if isinstance(first_arg, ast.Constant) and first_arg.value == "os.environ":
            return True

        # Check for os.environ as an expression
        resolved_arg = resolver.resolve(first_arg)
        if resolved_arg == ["os", "environ"]:
            return True

        return False
