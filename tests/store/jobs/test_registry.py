from unittest import mock

import pytest

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.store.jobs.registry import JobStoreRegistry
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore


@pytest.mark.parametrize(
    ("enable_workspaces", "store_cls"),
    [("false", SqlAlchemyJobStore), ("true", WorkspaceAwareSqlAlchemyJobStore)],
)
def test_builtin_job_store_for_database_uri(tmp_path, monkeypatch, enable_workspaces, store_cls):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, enable_workspaces)
    store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    store = JobStoreRegistry().get_store_builder(store_uri)(store_uri)

    assert type(store) is store_cls


def test_job_store_entrypoint_replaces_builtin_store_for_its_scheme():
    def plugin_builder(store_uri):
        pass

    entrypoint = mock.Mock(load=mock.Mock(return_value=plugin_builder))
    entrypoint.name = "postgresql"

    with mock.patch(
        "mlflow.utils.plugins._get_entry_points",
        side_effect=lambda group: [entrypoint] if group == "mlflow.job_store" else [],
    ):
        registry = JobStoreRegistry()

    assert registry.get_store_builder("postgresql://user@host:5432/mlflow") is plugin_builder
    assert registry.get_store_builder("postgresql+psycopg2://user@host/mlflow") is plugin_builder
    assert registry.get_store_builder("mysql://user@host/mlflow") is not plugin_builder
