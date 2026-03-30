import ast

from clint.rules.base import Rule


class ExceptBoolOp(Rule):
    def _message(self) -> str:
        return (
            "Did you mean `except (X, Y):`? Using or/and in an except handler is likely a mistake."
        )

    @staticmethod
    def check(node: ast.ExceptHandler) -> bool:
        return isinstance(node.type, ast.BoolOp)
