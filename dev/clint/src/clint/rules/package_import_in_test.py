import ast

from clint.builtin import BUILTIN_MODULES
from clint.rules.base import Rule


class PackageImportInTest(Rule):
    def _message(self) -> str:
        return (
            "Do not import packages within test functions. "
            "Import packages at the module level instead."
        )

    @staticmethod
    def check(node: ast.Import | ast.ImportFrom, is_in_test: bool) -> bool:
        """
        Returns True if this is a package import within a test function.

        Args:
            node: The import node to check.
            is_in_test: Whether the import is within a test function.

        Returns:
            True if this is a non-builtin package import within a test function.
        """
        if not is_in_test:
            return False

        # Get the module being imported
        if isinstance(node, ast.Import):
            # For "import foo" or "import foo.bar" or "import foo, bar"
            # Check all modules in the import statement
            for alias in node.names:
                module = alias.name.split(".", 1)[0]
                # If any non-builtin module is found, flag it
                if module not in BUILTIN_MODULES:
                    return True
            return False
        elif isinstance(node, ast.ImportFrom):
            # Check for relative imports first
            # (from . import, from .. import, from ...package import)
            if node.level > 0:
                # Relative imports like "from . import foo" or "from .. import bar"
                # or "from ...package import module"
                # These are fine - allow them
                return False
            # For "from foo import bar"
            elif node.module:
                module = node.module.split(".", 1)[0]
            else:
                # from __future__ import or similar - allow it
                return False
        else:
            return False

        # Allow builtin modules to be imported in tests
        if module in BUILTIN_MODULES:
            return False

        return True
