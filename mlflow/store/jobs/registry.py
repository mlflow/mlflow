from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking.registry import StoreRegistry


class JobStoreRegistry(StoreRegistry):
    """Scheme-based registry for job store implementations

    The built-in SQLAlchemy job store serves every database scheme. Packages can provide a
    job store for a scheme through the `mlflow.job_store` entrypoint group, which replaces
    the built-in store for that scheme. A store builder is called with the store URI.
    """

    def __init__(self):
        super().__init__("mlflow.job_store")
        for scheme in DATABASE_ENGINES:
            self.register(scheme, _get_sqlalchemy_job_store)
        self.register_entrypoints()


def _get_sqlalchemy_job_store(store_uri):
    from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
    from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore

    store_cls = (
        WorkspaceAwareSqlAlchemyJobStore if MLFLOW_ENABLE_WORKSPACES.get() else SqlAlchemyJobStore
    )
    return store_cls(store_uri)
