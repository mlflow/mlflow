import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class TempfileInTest(Rule):
    def _message(self) -> str:
        return (
            "Do not use `tempfile` in tests. Use the `tmp_path` fixture instead "
            "(https://docs.pytest.org/en/stable/reference/reference.html#tmp-path)."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        match resolver.resolve(node):
            case [
                "tempfile",
                "TemporaryDirectory" | "NamedTemporaryFile" | "TemporaryFile" | "mkdtemp",
            ]:
                return True
            case _:
                return False
