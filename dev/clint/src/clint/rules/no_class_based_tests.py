import ast

from typing_extensions import Self

from clint.rules.base import Rule


class NoClassBasedTests(Rule):
    def __init__(self, class_name: str) -> None:
        self.class_name = class_name

    @classmethod
    def check(cls, node: ast.ClassDef, path_name: str) -> Self | None:
        # Only check in test files
        if not path_name.startswith("test_"):
            return None

        if not node.name.startswith("Test"):
            return None

        # Check if the class has any test methods
        if any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name.startswith("test_")
            for stmt in node.body
        ):
            return cls(node.name)

        return None

    def _message(self) -> str:
        return (
            f"Class-based tests are not allowed. "
            f"Convert class '{self.class_name}' to function-based tests."
        )
