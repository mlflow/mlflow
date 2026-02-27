import ast

from clint.rules.base import Rule


class IncorrectTypeAnnotation(Rule):
    MAPPING = {
        "callable": "Callable",
        "any": "Any",
    }

    def __init__(self, type_hint: str) -> None:
        self.type_hint = type_hint

    @staticmethod
    def check(node: ast.Name) -> bool:
        return node.id in IncorrectTypeAnnotation.MAPPING

    def _message(self) -> str:
        if correct_hint := self.MAPPING.get(self.type_hint):
            return f"Did you mean `{correct_hint}` instead of `{self.type_hint}`?"

        raise ValueError(
            f"Unexpected type: {self.type_hint}. It must be one of {list(self.MAPPING)}."
        )
