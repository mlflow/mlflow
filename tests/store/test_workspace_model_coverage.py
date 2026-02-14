"""Verify that every SQLAlchemy model with a ``workspace`` column is handled
by at least one workspace store's ``_get_query`` method.

If a new model is added with a ``workspace`` column but the developer forgets
to add it to ``_get_query``, that model's queries will bypass workspace
isolation.  This test catches that gap.
"""

import ast
from pathlib import Path

# Explicitly import all dbmodel modules so that every ORM model is registered
# with Base.registry before we inspect mappers.
import mlflow.store.model_registry.dbmodels.models  # noqa: F401
import mlflow.store.tracking.dbmodels.models  # noqa: F401
import mlflow.store.workspace.dbmodels.models  # noqa: F401
from mlflow.store.db.base_sql_model import Base

# Locate the repository root relative to this test file so that workspace
# store paths resolve correctly regardless of the pytest working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Every workspace store that provides a ``_get_query`` override.
WORKSPACE_STORE_PATHS = [
    "mlflow/store/tracking/sqlalchemy_workspace_store.py",
    "mlflow/store/model_registry/sqlalchemy_workspace_store.py",
    "mlflow/store/jobs/sqlalchemy_workspace_store.py",
]


def _models_handled_by_get_query(ws_path: str) -> set[str]:
    """Parse a workspace store file and return the ``Sql*`` model names
    referenced in comparisons inside its ``_get_query`` method.
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
