"""Rule to detect redundant docstrings in test functions and classes.

This rule flags single-line docstrings in test functions and classes when:
1. The docstring is shorter than the function name, AND
2. More than 50% of the function name's words appear in the docstring

This catches docstrings that just restate the function name without adding
value. Multi-line docstrings are always allowed as they generally provide
meaningful context.
"""

import ast
import difflib
import re

from typing_extensions import Self

from clint.rules.base import Rule

MIN_WORD_OVERLAP_PERCENTAGE = 0.5
MAX_DOCSTRING_LENGTH_RATIO = 1.25


class RedundantTestDocstring(Rule):
    def __init__(
        self,
        function_name: str | None = None,
        has_class_docstring: bool = False,
        is_module_docstring: bool = False,
    ) -> None:
        self.function_name = function_name
        self.has_class_docstring = has_class_docstring
        self.is_module_docstring = is_module_docstring

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

        # Check if docstring exists and get the raw docstring for multiline detection
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.s, str)
        ):
            raw_docstring = node.body[0].value.s
            docstring = ast.get_docstring(node)

            # If raw docstring has newlines, it's multiline - always allow
            if "\n" in raw_docstring:
                return None

            if docstring and cls._is_redundant_docstring(docstring, node.name):
                return cls(node.name, has_class_docstring=is_class)

        return None

    @classmethod
    def check_module(cls, module: ast.Module, path_name: str) -> Self | None:
        """Check if module-level docstring is redundant."""
        if not (path_name.startswith("test_") or path_name.endswith("_test.py")):
            return None

        # Check raw docstring for multiline detection
        if (
            module.body
            and isinstance(module.body[0], ast.Expr)
            and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.s, str)
        ):
            raw_docstring = module.body[0].value.s
            # Only flag single-line module docstrings
            if "\n" not in raw_docstring:
                return cls(is_module_docstring=True)

        return None

    @staticmethod
    def _is_redundant_docstring(docstring: str, function_name: str) -> bool:
        """Check if a docstring is redundant based on length and similarity to function name."""
        stripped = docstring.strip()
        if len(stripped) > len(function_name) * MAX_DOCSTRING_LENGTH_RATIO:
            return False

        # Normalize function name: convert snake_case and camelCase to space-separated words
        func_normalized = function_name.replace("_", " ")
        func_normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", func_normalized)
        func_normalized = func_normalized.lower().replace("test", "").strip()
        # Remove punctuation and normalize whitespace
        func_normalized = re.sub(r"[^\w\s]", "", func_normalized)
        func_normalized = " ".join(func_normalized.split())

        # Normalize docstring
        doc_normalized = stripped.lower().replace("test", "").replace("tests", "").strip()
        # Remove punctuation and normalize whitespace
        doc_normalized = re.sub(r"[^\w\s]", "", doc_normalized)
        doc_normalized = " ".join(doc_normalized.split())

        if not func_normalized or not doc_normalized:
            return False

        # Use SequenceMatcher to check if docstring content appears in function name
        # This handles partial matches better (e.g., "validate" vs "validation")
        matcher = difflib.SequenceMatcher(None, doc_normalized, func_normalized)
        match = matcher.find_longest_match(0, len(doc_normalized), 0, len(func_normalized))

        # Calculate what percentage of the docstring matches the function name
        match_ratio = match.size / len(doc_normalized) if len(doc_normalized) > 0 else 0

        return match_ratio >= MIN_WORD_OVERLAP_PERCENTAGE

    def _message(self) -> str:
        if self.is_module_docstring:
            return (
                "Test module has a single-line docstring. "
                "Single-line module docstrings don't provide enough context. "
                "Consider removing it or expanding it with meaningful details."
            )

        entity_type = "Test class" if self.has_class_docstring else "Test function"
        return (
            f"{entity_type} '{self.function_name}' has a redundant docstring. "
            f"Short docstrings that restate the function name don't add value. "
            f"Consider removing it or expanding it with meaningful details."
        )
