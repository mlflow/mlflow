import ast

from typing_extensions import Self

from clint.resolver import Resolver
from clint.rules.base import Rule


class IncorrectTypeAnnotation(Rule):
    MAPPING = {
        "callable": "typing.Callable",
        "any": "typing.Any",
        "numpy.array": "numpy.ndarray",
        "dataclasses.dataclass": "Any",
    }

    def __init__(self, actual: str, expected: str) -> None:
        self.actual = actual
        self.expected = expected

    @classmethod
    def check(cls, node: ast.AST, resolver: Resolver) -> Self | None:
        names = resolver.resolve(node)
        actual = ".".join(names) if names else ast.unparse(node)
        if expected := cls.MAPPING.get(actual):
            return cls(actual=actual, expected=expected)

    def _message(self) -> str:
        if self.actual == "dataclasses.dataclass":
            return "`dataclass` is an invalid type annotation. Use `Any` instead as a workaround."
        else:
            return f"Did you mean `{self.expected}` instead of `{self.actual}`?"
