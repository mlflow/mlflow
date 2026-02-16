import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class UnparameterizedGenericType(Rule):
    def __init__(self, type_hint: str) -> None:
        self.type_hint = type_hint

    @staticmethod
    def is_generic_type(node: ast.Name | ast.Attribute, resolver: Resolver) -> bool:
        if names := resolver.resolve(node):
            return tuple(names) in {
                ("typing", "Callable"),
                ("typing", "Sequence"),
            }
        elif isinstance(node, ast.Name):
            return node.id in {
                "dict",
                "list",
                "set",
                "tuple",
                "frozenset",
            }
        return False

    def _message(self) -> str:
        return (
            f"Generic type `{self.type_hint}` must be parameterized "
            "(e.g., `list[str]` rather than `list`)."
        )
