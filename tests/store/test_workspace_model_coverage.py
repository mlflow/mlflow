"""Verify that every SQLAlchemy model with a ``workspace`` column is handled
by at least one workspace store's ``_get_query`` method.

If a new model is added with a ``workspace`` column but the developer forgets
to add it to ``_get_query``, that model's queries will bypass workspace
isolation.  This test catches that gap.
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

# Locate the repository root relative to this test file so that workspace
# store paths resolve correctly regardless of the pytest working directory.
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


def _models_handled_by_get_query(ws_path: str) -> set[str]:
    """Parse a workspace store file and return the ``Sql*`` model names
    referenced in comparisons inside its ``_get_query`` method.

    Note: this deliberately duplicates logic from the clint linter rule
    (``_get_workspace_isolated_models``) so that this test has no dependency
    on the ``clint`` package, which is not in the standard test requirements.
    """
    tree = ast.parse((_REPO_ROOT / ws_path).read_text())

    # Collect module-level variables holding Sql* names (e.g. tuples of models)
    var_models: dict[str, set[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names = {
                        n.id
                        for n in ast.walk(node.value)
                        if isinstance(n, ast.Name) and n.id.startswith("Sql")
                    }
                    if names:
                        var_models[target.id] = names

    # Extract model names from _get_query comparisons
    models: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "_get_query":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Compare):
                continue
            for comp in inner.comparators:
                models |= {
                    n.id
                    for n in ast.walk(comp)
                    if isinstance(n, ast.Name) and n.id.startswith("Sql")
                }
                if isinstance(comp, ast.Name) and comp.id in var_models:
                    models |= var_models[comp.id]
    return models


def test_all_workspace_models_handled_in_get_query():
    """Every model with a workspace column must appear in at least one
    workspace store's ``_get_query``.
    """
    handled: set[str] = set()
    for ws_path in WORKSPACE_STORE_PATHS:
        handled |= _models_handled_by_get_query(ws_path)

    models_with_column: set[str] = set()
    for mapper in Base.registry.mappers:
        if "workspace" in {col.key for col in mapper.columns}:
            models_with_column.add(mapper.class_.__name__)

    missing = models_with_column - handled
    assert not missing, (
        f"These models have a `workspace` column but are not handled by any "
        f"workspace store's _get_query: {sorted(missing)}. "
        f"Add handling in the appropriate workspace store's _get_query method."
    )


def _build_model_children() -> dict[str, set[str]]:
    """Build a parent -> children map from FK relationships.

    For each workspace-isolated model (handled by ``_get_query``), determine
    which other workspace-isolated models are reachable via FK chains.  For
    example, ``SqlRun`` has an ``experiment_id`` FK to ``SqlExperiment``, so
    ``SqlRun`` is a child of ``SqlExperiment``.
    """
    # All models handled by _get_query across all workspace stores
    isolated: set[str] = set()
    for ws_path in WORKSPACE_STORE_PATHS:
        isolated |= _models_handled_by_get_query(ws_path)

    # Map table name -> class name for FK target resolution
    table_to_class: dict[str, str] = {}
    class_to_mapper = {}
    for mapper in Base.registry.mappers:
        table_to_class[mapper.persist_selectable.name] = mapper.class_.__name__
        class_to_mapper[mapper.class_.__name__] = mapper

    # For each isolated model, find its FK parents (also isolated)
    children: dict[str, set[str]] = {}
    for model_name in isolated:
        mapper = class_to_mapper.get(model_name)
        if mapper is None:
            continue
        for col in mapper.columns:
            for fk in col.foreign_keys:
                parent_name = table_to_class.get(fk.column.table.name)
                if parent_name and parent_name in isolated and parent_name != model_name:
                    children.setdefault(parent_name, set()).add(model_name)

    return children


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
    least one workspace store class.

    If this test fails, a HELPER_MODEL_COVERAGE entry references a method that
    doesn't exist (typo or removed method).
    """
    coverage_keys = _parse_helper_model_coverage_keys_from_source()
    ws_methods = _all_workspace_store_methods()
    # Also include base store public methods that are workspace-safe callers
    # (e.g. get_experiment) -- these aren't defined in workspace stores but are
    # recognized by the linter.
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


def _parse_model_children_from_source() -> dict[str, set[str]]:
    """Parse MODEL_CHILDREN from the linter source without importing clint."""
    src = _LINTER_RULE_PATH.read_text()
    tree = ast.parse(src)
    result: dict[str, set[str]] = {}
    for node in ast.iter_child_nodes(tree):
        # MODEL_CHILDREN uses a type annotation, so it's AnnAssign not Assign.
        if not isinstance(node, ast.AnnAssign):
            continue
        if not (isinstance(node.target, ast.Name) and node.target.id == "MODEL_CHILDREN"):
            continue
        # MODEL_CHILDREN is a dict literal: {str: frozenset({str, ...}), ...}
        if not isinstance(node.value, ast.Dict):
            break
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                children: set[str] = set()
                for inner in ast.walk(value):
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                        children.add(inner.value)
                result[key.value] = children
        break
    return result


@pytest.mark.skipif(
    not _LINTER_RULE_PATH.exists(),
    reason="clint linter source not available in this environment",
)
def test_model_children_matches_fk_relationships():
    """Verify the static MODEL_CHILDREN dict in the linter matches the actual
    FK relationships between workspace-isolated models.

    If this test fails, update MODEL_CHILDREN in
    ``dev/clint/src/clint/rules/workspace_query_isolation.py`` to match.
    """
    expected = _build_model_children()
    actual = _parse_model_children_from_source()

    # Check for missing parents
    for parent in expected:
        if parent not in actual:
            assert False, (
                f"MODEL_CHILDREN is missing parent {parent!r} with children "
                f"{sorted(expected[parent])}. Add it to MODEL_CHILDREN."
            )

    # Check for missing/extra children per parent
    for parent in expected:
        missing = expected[parent] - actual.get(parent, set())
        assert not missing, (
            f"MODEL_CHILDREN[{parent!r}] is missing children: {sorted(missing)}. "
            f"Expected: {sorted(expected[parent])}"
        )

    # Check for stale entries in MODEL_CHILDREN
    for parent in actual:
        if parent not in expected:
            extra = actual[parent]
            assert False, (
                f"MODEL_CHILDREN has stale parent {parent!r} with children "
                f"{sorted(extra)} but no FK relationships exist."
            )
        extra = actual[parent] - expected.get(parent, set())
        assert not extra, (
            f"MODEL_CHILDREN[{parent!r}] has extra children with no FK: {sorted(extra)}"
        )
