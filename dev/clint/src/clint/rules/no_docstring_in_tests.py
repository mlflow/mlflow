"""Rule to enforce self-documenting test function names without docstrings.

This rule ensures that test functions and classes use descriptive names
instead of docstrings. For critical documentation within tests, use
NB (nota bene) comments:

    def test_complex_edge_case():
        # NB: This test uses a workaround for issue #12345 because
        # the standard approach fails due to timing constraints.
        special_setup()
        assert result == expected
"""

import ast

from typing_extensions import Self

from clint.rules.base import Rule


class NoDocstringInTests(Rule):
    def __init__(self, function_name: str, has_class_docstring: bool = False) -> None:
        self.function_name = function_name
        self.has_class_docstring = has_class_docstring

    @classmethod
    def check(
        cls, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, path_name: str
    ) -> Self | None:
        # Only check test files
        if not path_name.startswith("test_") and not path_name.endswith("_test.py"):
            return None

        # Skip conftest.py files as they often have utility functions with docstrings
        if path_name == "conftest.py":
            return None

        # Check if this is a test function or test class
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if it's a test function
            if not node.name.startswith("test_"):
                return None

            # Check if the function has a docstring
            if ast.get_docstring(node):
                return cls(node.name)

        elif isinstance(node, ast.ClassDef):
            # Check if it's a test class
            if not node.name.startswith("Test"):
                return None

            # Check if the class has a docstring
            if ast.get_docstring(node):
                return cls(node.name, has_class_docstring=True)

        return None

    def _message(self) -> str:
        if self.has_class_docstring:
            return (
                f"Test class '{self.function_name}' should not have a docstring. "
                f"Test class names should be self-documenting. "
                f"Use NB comments for critical documentation."
            )
        return (
            f"Test function '{self.function_name}' should not have a docstring. "
            f"Test function names should be self-documenting. "
            f"Use NB comments for critical documentation."
        )
