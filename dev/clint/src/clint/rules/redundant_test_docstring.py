"""Rule to detect redundant docstrings in test functions and classes.

This rule flags ALL single-line docstrings in test functions and classes.
Single-line docstrings in tests rarely provide meaningful context and are
typically redundant. Multi-line docstrings are always allowed as they
generally provide meaningful context.
"""

import ast

from typing_extensions import Self

from clint.rules.base import Rule


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

            # If raw docstring has newlines, it's multiline - always allow
            if "\n" in raw_docstring:
                return None

            # Single-line docstrings in test functions/classes rarely provide meaningful context
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

    def _message(self) -> str:
        if self.is_module_docstring:
            return (
                "Test module has a single-line docstring. "
                "Single-line module docstrings don't provide enough context. "
                "Consider removing it."
            )

        entity_type = "Test class" if self.has_class_docstring else "Test function"
        return (
            f"{entity_type} '{self.function_name}' has a single-line docstring. "
            f"Single-line docstrings in tests rarely provide meaningful context. "
            f"Consider removing it."
        )
