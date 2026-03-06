"""Verify that every SQLAlchemy model with a ``workspace`` column subclasses
``WorkspaceIsolatedModel`` and that every ``WorkspaceIsolatedModel`` subclass
implements ``workspace_query_filter``.
"""

import ast
from pathlib import Path

import pytest

# Explicitly import all dbmodel modules so that every ORM model is registered
# with Base.registry before we inspect mappers.
import mlflow.store.model_registry.dbmodels.models  # noqa: F401
import mlflow.store.tracking.dbmodels.models  # noqa: F401
import mlflow.store.workspace.dbmodels.models  # noqa: F401
from mlflow.store.db.base_sql_model import Base
from mlflow.store.db.workspace_isolated_model import WorkspaceIsolatedModel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The linter source lives under dev/clint/ and may not be present in all
# environments (e.g., packaged distributions or minimal CI checkouts).
_LINTER_RULE_PATH = _REPO_ROOT / "dev/clint/src/clint/rules/workspace_query_isolation.py"

# Every workspace store that provides a ``_get_query`` override.
WORKSPACE_STORE_PATHS = [
    "mlflow/store/tracking/sqlalchemy_workspace_store.py",
    "mlflow/store/model_registry/sqlalchemy_workspace_store.py",
    "mlflow/store/jobs/sqlalchemy_workspace_store.py",
]


def test_workspace_column_requires_mixin():
    """Every model with a ``workspace`` column must subclass
    ``WorkspaceIsolatedModel``.
    """
    missing = []
    for mapper in Base.registry.mappers:
        cls = mapper.class_
        if "workspace" in {col.key for col in mapper.columns}:
            if not issubclass(cls, WorkspaceIsolatedModel):
                missing.append(cls.__name__)
    assert not missing, (
        f"These models have a `workspace` column but do not subclass "
        f"WorkspaceIsolatedModel: {sorted(missing)}. "
        f"Add WorkspaceIsolatedModel to their bases and implement "
        f"workspace_query_filter."
    )


def test_all_workspace_isolated_models_implement_filter():
    """Every ``WorkspaceIsolatedModel`` subclass must define
    ``workspace_query_filter`` in its own class (not just inherit it).
    """
    missing = []
    for mapper in Base.registry.mappers:
        cls = mapper.class_
        if issubclass(cls, WorkspaceIsolatedModel) and cls is not WorkspaceIsolatedModel:
            if "workspace_query_filter" not in cls.__dict__:
                missing.append(cls.__name__)
    assert not missing, (
        f"These WorkspaceIsolatedModel subclasses do not define "
        f"workspace_query_filter: {sorted(missing)}. "
        f"Each model must implement its own workspace_query_filter classmethod."
    )


def _parse_helper_model_coverage_keys_from_source() -> set[str]:
    """Parse HELPER_MODEL_COVERAGE keys from the linter source without importing clint."""
    src = _LINTER_RULE_PATH.read_text()
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not (isinstance(node.target, ast.Name) and node.target.id == "HELPER_MODEL_COVERAGE"):
            continue
        if isinstance(node.value, ast.Dict):
            keys = {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if not keys:
                raise AssertionError(
                    f"Failed to parse any HELPER_MODEL_COVERAGE keys from {_LINTER_RULE_PATH}"
                )
            return keys
    raise AssertionError(
        f"HELPER_MODEL_COVERAGE annotated assignment not found in {_LINTER_RULE_PATH}"
    )


def _all_workspace_store_methods() -> set[str]:
    """Collect all method names defined in any workspace store class."""
    methods: set[str] = set()
    for ws_path in WORKSPACE_STORE_PATHS:
        tree = ast.parse((_REPO_ROOT / ws_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
    return methods


@pytest.mark.skipif(
    not _LINTER_RULE_PATH.exists(),
    reason="clint linter source not available in this environment",
)
def test_helper_model_coverage_entries_are_real_methods():
    """Verify every key in HELPER_MODEL_COVERAGE is defined as a method in at
    least one workspace store class or base store class.
    """
    coverage_keys = _parse_helper_model_coverage_keys_from_source()
    ws_methods = _all_workspace_store_methods()
    base_store_paths = [
        "mlflow/store/tracking/sqlalchemy_store.py",
        "mlflow/store/model_registry/sqlalchemy_store.py",
        "mlflow/store/jobs/sqlalchemy_store.py",
    ]
    for base_path in base_store_paths:
        tree = ast.parse((_REPO_ROOT / base_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ws_methods.add(item.name)

    unknown = coverage_keys - ws_methods
    assert not unknown, (
        f"HELPER_MODEL_COVERAGE references methods not found in any store: "
        f"{sorted(unknown)}. Remove stale entries or fix typos."
    )
