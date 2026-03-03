import ast
from typing import Literal

from typing_extensions import Self

from clint.resolver import Resolver
from clint.rules.base import Rule


# See https://github.com/astral-sh/ruff/issues/3608
class PreferOsEnviron(Rule):
    def __init__(self, func: Literal["getenv", "putenv"]) -> None:
        self.func = func

    def _message(self) -> str:
        if self.func == "putenv":
            return "Use `os.environ[key] = value` instead of `os.putenv()`."
        return "Use `os.environ.get()` instead of `os.getenv()`."

    @classmethod
    def check(cls, node: ast.Call, resolver: Resolver) -> Self | None:
        match resolver.resolve(node.func):
            case ["os", ("getenv" | "putenv") as func]:
                return cls(func)
        return None
