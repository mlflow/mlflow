import importlib
import io
import json
from unittest import mock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import mysql

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import SqlIssue
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


def test_issue_metadata_mysql_schema_capacity():
    migration = importlib.import_module(
        "mlflow.store.db_migrations.versions.8b6ad64cb8f5_extend_issue_contract"
    )
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="mysql", opts={"as_sql": True, "output_buffer": output}
    )
    connection = mock.Mock()
    connection.execute.return_value.fetchmany.return_value = []
    with (
        Operations.context(context),
        mock.patch.object(migration.op, "get_bind", return_value=connection) as get_bind,
    ):
        migration.upgrade()
        get_bind.assert_called_once()
    assert "ADD COLUMN metadata LONGTEXT" in output.getvalue()
    assert SqlIssue.__table__.c.metadata.type.compile(dialect=mysql.dialect()) == "LONGTEXT"


@pytest.mark.parametrize("previous_revision", ["dc11669786a5", "e7d1f4b2a9c6"])
def test_issue_contract_migration_preserves_existing_issues(tmp_path, previous_revision):
    db_url = f"sqlite:///{tmp_path / 'issue-migration.db'}"
    engine = sa.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    config = _get_alembic_config(db_url)
    command.upgrade(config, previous_revision)
    tables = sa.MetaData()
    tables.reflect(engine)
    with engine.begin() as conn:
        conn.execute(
            tables
            .tables["experiments"]
            .insert()
            .values(
                experiment_id=1,
                name="legacy",
                artifact_location="",
                lifecycle_stage="active",
                workspace="default",
            )
        )
        conn.execute(
            tables
            .tables["runs"]
            .insert()
            .values(run_uuid="old-run", experiment_id=1, lifecycle_stage="active")
        )
        conn.execute(
            tables
            .tables["issues"]
            .insert()
            .values(
                issue_id="iss-old",
                experiment_id=1,
                name="Legacy issue",
                description="Description",
                status="pending",
                severity=None,
                root_causes=json.dumps(["Old cause"]),
                categories=json.dumps(["cost", "cost", "c" * 300]),
                created_timestamp=100,
                last_updated_timestamp=200,
                source_run_id="old-run",
                created_by="author",
            )
        )
    command.upgrade(config, "8b6ad64cb8f5")
    store = SqlAlchemyStore(db_url, str(tmp_path / "artifacts"))
    issue = store.get_issue("iss-old")
    assert issue.last_seen_timestamp == 100
    assert issue.last_updated_timestamp == 200
    assert issue.last_updated_by is None
    assert issue.root_causes == ["Old cause"]
    assert issue.categories == ["cost", "cost", "c" * 300]
    assert issue.source_run_id == "old-run"
    assert issue.severity is None
    assert [result.issue_id for result in store.search_issues(categories=["cost"])] == ["iss-old"]
    assert store.update_issue("iss-old", categories=["latency"]).categories == ["latency"]
    assert store.search_issues(categories=["cost"]) == []
