"""Verify that every SQLAlchemy model with a ``workspace`` column uses the
``WorkspaceIsolatedModel`` mixin and implements ``workspace_query_filter``.
"""

# Explicitly import all dbmodel modules so that every ORM model is registered
# with Base.registry before we inspect mappers.
import mlflow.store.model_registry.dbmodels.models  # noqa: F401
import mlflow.store.tracking.dbmodels.models  # noqa: F401
import mlflow.store.workspace.dbmodels.models  # noqa: F401
from mlflow.store.db.base_sql_model import Base
from mlflow.store.db.workspace_isolated_model import WorkspaceIsolatedModel


def test_workspace_column_requires_mixin():
    for mapper in Base.registry.mappers:
        model = mapper.class_
        columns = {col.key for col in mapper.columns}
        if "workspace" in columns:
            assert issubclass(model, WorkspaceIsolatedModel), (
                f"{model.__name__} has a workspace column but does not "
                f"inherit from WorkspaceIsolatedModel"
            )


def test_all_workspace_isolated_models_implement_filter():
    for mapper in Base.registry.mappers:
        model = mapper.class_
        if issubclass(model, WorkspaceIsolatedModel):
            assert "workspace_query_filter" in model.__dict__, (
                f"{model.__name__} inherits WorkspaceIsolatedModel but does not "
                f"implement workspace_query_filter"
            )
