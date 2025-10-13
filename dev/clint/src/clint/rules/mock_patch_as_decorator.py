import ast
from pathlib import Path

from clint.resolver import Resolver
from clint.rules.base import Rule


class MockPatchAsDecorator(Rule):
    # TODO: Gradually migrate these files to use mock.patch as context manager
    # Remove files from this list once they've been migrated
    # Files are sorted by violation count (descending) to prioritize migration
    IGNORED_FILES = {
        "tests/utils/test_databricks_utils.py",  # 10
        "tests/genai/scorers/test_scorer_CRUD.py",  # 5
        "tests/store/tracking/test_databricks_rest_store.py",  # 4
    }

    def _message(self) -> str:
        return (
            "Do not use `unittest.mock.patch` as a decorator. "
            "Use it as a context manager to avoid patches being active longer than needed "
            "and to make it clear which code depends on them."
        )

    @staticmethod
    def check(
        decorator_list: list[ast.expr], resolver: Resolver, file_path: Path | None = None
    ) -> ast.expr | None:
        """
        Returns the decorator node if it is a `@mock.patch` or `@patch` decorator.
        """
        # Skip files in the ignore list
        if file_path and str(file_path) in MockPatchAsDecorator.IGNORED_FILES:
            return None

        for deco in decorator_list:
            if res := resolver.resolve(deco):
                match res:
                    # Resolver returns ["unittest", "mock", "patch", ...]
                    # The *_ captures variants like "object", "dict", etc.
                    case ["unittest", "mock", "patch", *_]:
                        return deco
        return None
