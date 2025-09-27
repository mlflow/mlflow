"""Rule to detect redundant docstrings in test functions and classes.

This rule flags single-line docstrings in test functions and classes when:
1. The docstring is shorter than the function name, AND
2. More than 50% of the function name's words appear in the docstring

This catches docstrings that just restate the function name without adding
value. Multi-line docstrings are always allowed as they generally provide
meaningful context.
"""

import ast
import re

from typing_extensions import Self

from clint.rules.base import Rule

MIN_WORD_OVERLAP_PERCENTAGE = 0.5
MAX_DOCSTRING_LENGTH_RATIO = 1.0


class RedundantTestDocstring(Rule):
    def __init__(self, function_name: str, has_class_docstring: bool = False) -> None:
        self.function_name = function_name
        self.has_class_docstring = has_class_docstring

    @classmethod
    def check(
        cls, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, path_name: str
    ) -> Self | None:
        if not (path_name.startswith("test_") or path_name.endswith("_test.py")):
            return None

        is_class = isinstance(node, ast.ClassDef)

        if is_class and not node.name.startswith("Test"):
            return None
        if not is_class and not node.name.startswith("test_"):
            return None

        if (docstring := ast.get_docstring(node)) and cls._is_redundant_docstring(
            docstring, node.name, cls._get_raw_docstring(node)
        ):
            return cls(node.name, has_class_docstring=is_class)

        return None

    @staticmethod
    def _get_raw_docstring(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> str | None:
        """Get the raw docstring value from a node."""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    @staticmethod
    def _is_redundant_docstring(
        docstring: str, function_name: str, raw_docstring: str | None
    ) -> bool:
        """Check if a docstring is redundant based on length and word overlap with function name."""
        if raw_docstring and "\n" in raw_docstring:
            return False

        stripped = docstring.strip()
        if len(stripped) > len(function_name) * MAX_DOCSTRING_LENGTH_RATIO:
            return False

        func_words = (
            set(function_name.lower().split("_"))
            if "_" in function_name
            else {word.lower() for word in re.findall(r"[A-Z][a-z]*|[a-z]+", function_name)}
        )
        func_words -= {"test", ""}

        doc_words = set(re.findall(r"\b[a-z]+\b", stripped.lower())) - {"test", "tests"}

        if not func_words or not doc_words:
            return False

        overlap_percentage = len(func_words & doc_words) / len(func_words)
        return overlap_percentage >= MIN_WORD_OVERLAP_PERCENTAGE

    def _message(self) -> str:
        entity_type = "Test class" if self.has_class_docstring else "Test function"
        return (
            f"{entity_type} '{self.function_name}' has a redundant docstring. "
            f"Short docstrings that restate the function name don't add value. "
            f"Consider removing it or expanding it with meaningful details."
        )
