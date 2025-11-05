import ast

from clint.rules.base import Rule


class PackageImportInTest(Rule):
    def _message(self) -> str:
        return (
            "Do not import modules within test functions. "
            "Import modules at the module level instead."
        )

    @staticmethod
    def check(node: ast.Import | ast.ImportFrom, is_in_test: bool) -> bool:
        """
        Returns True if this is an import within a test function.

        Args:
            node: The import node to check.
            is_in_test: Whether the import is within a test function.

        Returns:
            True if this is any import within a test function.
        """
        # Flag all imports within test functions
        return is_in_test
