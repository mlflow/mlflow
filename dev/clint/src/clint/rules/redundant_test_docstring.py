"""Rule to prevent single-line docstrings in test functions and classes.

This rule ensures that test functions and classes don't have single-line docstrings,
which are typically low-value like "Test X" or "Tests for Y". Multi-line docstrings
that provide substantial documentation are allowed.

For critical inline documentation, NB (nota bene) comments are encouraged:

    def test_complex_edge_case():
        # NB: This test uses a workaround for issue #12345 because
        # the standard approach fails due to timing constraints.
        special_setup()
        assert result == expected
"""

import ast

from typing_extensions import Self

from clint.rules.base import Rule


class RedundantTestDocstring(Rule):
    def __init__(self, function_name: str, has_class_docstring: bool = False) -> None:
        self.function_name = function_name
        self.has_class_docstring = has_class_docstring

    @classmethod
    def check(
        cls, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, path_name: str
    ) -> Self | None:
        if not path_name.startswith("test_") and not path_name.endswith("_test.py"):
            return None

        if path_name == "conftest.py":
            return None

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                return None

            docstring = ast.get_docstring(node)
            if docstring is not None and cls._is_single_line_docstring(docstring):
                return cls(node.name)

        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("Test"):
                return None

            docstring = ast.get_docstring(node)
            if docstring is not None and cls._is_single_line_docstring(docstring):
                return cls(node.name, has_class_docstring=True)

        return None

    @staticmethod
    def _is_single_line_docstring(docstring: str) -> bool:
        """Check if a docstring is a single line (after stripping whitespace)."""
        return docstring.strip().count("\n") == 0

    def _message(self) -> str:
        entity_type = "Test class" if self.has_class_docstring else "Test function"
        return (
            f"{entity_type} '{self.function_name}' has a single-line docstring. "
            f"Single-line docstrings in tests are typically low-value. "
            f"Either remove it or update the test name to be more descriptive."
        )
