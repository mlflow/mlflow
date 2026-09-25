import os
import uuid

import pytest
import sqlalchemy as sa

from mlflow.entities import ViewType
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.sqlalchemy_workspace_store import (
    WorkspaceAwareSqlAlchemyStore as WorkspaceAwareRegistryStore,
)
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock

DB_URI = os.environ.get("MLFLOW_TRACKING_URI")


def _psycopg3_uri():
    if not DB_URI or not DB_URI.startswith("postgresql"):
        pytest.skip("Only PostgreSQL rejects comparing an integer column to a string parameter")
    pytest.importorskip("psycopg")
    return (
        sa
        .make_url(DB_URI)
        .set(drivername="postgresql+psycopg")
        .render_as_string(hide_password=False)
    )


@pytest.fixture
def psycopg3_store(tmp_path, monkeypatch):
    """
    Workspace-aware store bound to the psycopg3 driver.

    psycopg3 binds parameters server-side and renders their type (`$1::VARCHAR`), so PostgreSQL
    rejects a string compared against an INTEGER column. psycopg2, the driver the other db tests
    run on, interpolates the parameter as an untyped literal that PostgreSQL coerces for us.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = WorkspaceAwareSqlAlchemyStore(_psycopg3_uri(), artifact_dir.as_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


@pytest.fixture
def psycopg3_registry_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareRegistryStore(_psycopg3_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


def test_experiment_id_filters_bind_integers(psycopg3_store):
    # Both call paths pass string experiment ids into a filter on the INTEGER
    # `experiments.experiment_id` column, which fails with "operator does not exist:
    # integer = character varying" unless they are coerced first.
    # https://github.com/mlflow/mlflow/issues/25188
    with WorkspaceContext("team-a"):
        exp_id = psycopg3_store.create_experiment(f"filters-{uuid.uuid4().hex}")

        traces, _ = psycopg3_store.search_traces(experiment_ids=[exp_id])
        assert traces == []

        associations = psycopg3_store.search_entities_by_destination(
            destination_ids=exp_id,
            destination_type=EntityAssociationType.EXPERIMENT,
            source_type=EntityAssociationType.EVALUATION_DATASET,
        )
        assert associations.to_list() == []


def test_search_experiments_experiment_id_filter_binds_integers(psycopg3_store):
    # `experiment_id = ...` / `IN (...)` filters bind against the INTEGER
    # `experiments.experiment_id` column; without coercing the filter value to
    # int first, psycopg v3's typed VARCHAR bind fails with "operator does not
    # exist: integer = character varying".
    with WorkspaceContext("team-a"):
        exp_id = psycopg3_store.create_experiment(f"filter-{uuid.uuid4().hex}")

        results = psycopg3_store.search_experiments(filter_string=f"experiment_id = '{exp_id}'")
        assert {e.experiment_id for e in results} == {exp_id}

        results = psycopg3_store.search_experiments(filter_string=f"experiment_id IN ('{exp_id}')")
        assert {e.experiment_id for e in results} == {exp_id}

        with pytest.raises(MlflowException, match="must be a valid integer"):
            psycopg3_store.search_experiments(filter_string="experiment_id = 'not-a-number'")


def test_search_experiments_time_filter_binds_integers(psycopg3_store):
    with WorkspaceContext("team-a"):
        exp_id = psycopg3_store.create_experiment(f"filter-{uuid.uuid4().hex}")
        experiment = psycopg3_store.get_experiment(exp_id)

        results = psycopg3_store.search_experiments(
            filter_string=f"creation_time = {experiment.creation_time}"
        )

        assert exp_id in {experiment.experiment_id for experiment in results}


def test_search_datasets_time_filter_binds_integers(psycopg3_store):
    with WorkspaceContext("team-a"):
        dataset = psycopg3_store.create_dataset(f"filter-{uuid.uuid4().hex}")

        results = psycopg3_store.search_datasets(
            filter_string=f"created_time = {dataset.created_time}"
        )

        assert dataset.dataset_id in {dataset.dataset_id for dataset in results}


def test_search_runs_time_filter_binds_integers(psycopg3_store):
    with WorkspaceContext("team-a"):
        exp_id = psycopg3_store.create_experiment(f"filter-{uuid.uuid4().hex}")
        run = psycopg3_store.create_run(
            exp_id,
            user_id="test-user",
            start_time=1234,
            tags=[],
            run_name="numeric-filter",
        )

        results = psycopg3_store.search_runs(
            [exp_id],
            filter_string="attributes.start_time = 1234",
            run_view_type=ViewType.ALL,
        )

        assert [result.info.run_id for result in results] == [run.info.run_id]


def test_search_mcp_registry_time_filters_bind_integers(psycopg3_store):
    with WorkspaceContext("team-a"):
        name = f"io.github.test/server-{uuid.uuid4().hex}"
        version = psycopg3_store.create_mcp_server_version({
            "name": name,
            "version": "1.0.0",
            "title": "Test server",
        })
        endpoint = psycopg3_store.create_mcp_access_endpoint(
            server_name=name,
            url="https://example.com/mcp",
            server_version=version.version,
        )

        servers = psycopg3_store.search_mcp_servers(
            filter_string=f"name = '{name}' AND created_at > 0"
        )
        versions = psycopg3_store.search_mcp_server_versions(
            name,
            filter_string="created_at > 0",
        )
        endpoints = psycopg3_store.search_mcp_access_endpoints(
            server_name=name,
            filter_string="created_at > 0",
        )

        assert [server.name for server in servers] == [name]
        assert [result.version for result in versions] == [version.version]
        assert [result.id for result in endpoints] == [endpoint.id]


def test_search_model_versions_version_number_filter_binds_integers(psycopg3_registry_store):
    with WorkspaceContext("team-a"):
        name = f"model-{uuid.uuid4().hex}"
        psycopg3_registry_store.create_registered_model(name)
        psycopg3_registry_store.create_model_version(name, "source")
        version = psycopg3_registry_store.create_model_version(name, "source")

        results = psycopg3_registry_store.search_model_versions(
            filter_string=f"name = '{name}' AND version_number = '{version.version}'"
        )

        assert [mv.version for mv in results] == [version.version]
