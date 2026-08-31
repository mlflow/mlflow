import os
import uuid

import pytest
import sqlalchemy as sa

from mlflow.entities.entity_type import EntityAssociationType
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock

DB_URI = os.environ.get("MLFLOW_TRACKING_URI")


@pytest.fixture
def psycopg3_store(tmp_path, monkeypatch):
    """
    Workspace-aware store bound to the psycopg3 driver.

    psycopg3 binds parameters server-side and renders their type (`$1::VARCHAR`), so PostgreSQL
    rejects a string compared against an INTEGER column. psycopg2, the driver the other db tests
    run on, interpolates the parameter as an untyped literal that PostgreSQL coerces for us.
    """
    if not DB_URI or not DB_URI.startswith("postgresql"):
        pytest.skip("Only PostgreSQL rejects comparing an integer column to a string parameter")
    pytest.importorskip("psycopg")

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    psycopg3_uri = (
        sa
        .make_url(DB_URI)
        .set(drivername="postgresql+psycopg")
        .render_as_string(hide_password=False)
    )
    store = WorkspaceAwareSqlAlchemyStore(psycopg3_uri, artifact_dir.as_uri())
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
