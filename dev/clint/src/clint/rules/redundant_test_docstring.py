"""Rule to detect redundant docstrings in test files.

This rule flags:
- ALL single-line docstrings in test functions and classes (multi-line
  function/class docstrings are allowed since they generally provide
  meaningful context).
- ALL module-level docstrings in test files (single- or multi-line).
"""

import ast

from clint.rules.base import Rule


class RedundantTestDocstring(Rule):
    @staticmethod
    def check(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, path_name: str
    ) -> ast.Constant | None:
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
            and isinstance(node.body[0].value.value, str)
        ):
            raw_docstring = node.body[0].value.value

            # If raw docstring has newlines, it's multiline - always allow
            if "\n" in raw_docstring:
                return None

            # Return the docstring node to flag
            return node.body[0].value

        return None

    @staticmethod
    def check_module(module: ast.Module, path_name: str) -> ast.Constant | None:
        if not (path_name.startswith("test_") or path_name.endswith("_test.py")):
            return None

        if (
            module.body
            and isinstance(module.body[0], ast.Expr)
            and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.value, str)
        ):
            return module.body[0].value

        return None

    def _message(self) -> str:
        return "Docstrings in test files rarely provide meaningful context. Consider removing it."
