import ast
from pathlib import Path

from clint.resolver import Resolver
from clint.rules.base import Rule


class MockPatchAsDecorator(Rule):
    # TODO: Gradually migrate these files to use mock.patch as context manager
    # Remove files from this list once they've been migrated (total: 92 violations)
    # Files are sorted by violation count (descending) to prioritize migration
    IGNORED_FILES = {
        "tests/utils/test_rest_utils.py",  # 15
        "tests/store/artifact/test_hdfs_artifact_repo.py",  # 10
        "tests/utils/test_databricks_utils.py",  # 10
        "tests/genai/scorers/test_builtin_scorers.py",  # 6
        "tests/evaluate/test_evaluation.py",  # 5
        "tests/genai/scorers/test_scorer_CRUD.py",  # 5
        "tests/langchain/test_langchain_model_export.py",  # 5
        "tests/projects/test_docker_projects.py",  # 4
        "tests/store/tracking/test_databricks_rest_store.py",  # 4
        "tests/store/tracking/test_rest_store.py",  # 4
        "tests/store/tracking/test_sqlalchemy_store.py",  # 4
        "tests/genai/test_genai_import_without_agent_sdk.py",  # 3
        "tests/mistral/test_mistral_autolog.py",  # 3
        "tests/models/test_model_config.py",  # 3
        "tests/transformers/test_transformers_model_export.py",  # 3
        "tests/genai/judges/test_judge_utils.py",  # 2
        "tests/projects/test_databricks.py",  # 2
        "tests/store/_unity_catalog/model_registry/test_unity_catalog_rest_store.py",  # 2
        "tests/tracing/utils/test_processor.py",  # 2
        "tests/autologging/test_autologging_safety_unit.py",  # 1
        "tests/db/test_tracking_operations.py",  # 1
        "tests/evaluate/logging/test_evaluation.py",  # 1
        "tests/genai/evaluate/test_context.py",  # 1
        "tests/genai/evaluate/test_evaluation.py",  # 1
        "tests/pyfunc/test_pyfunc_model_with_type_hints.py",  # 1
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
