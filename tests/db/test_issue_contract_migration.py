import importlib
import io
import json
import uuid
from unittest import mock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import mysql

from mlflow.entities import TraceInfo, TraceState
from mlflow.entities.assessment import IssueReference
from mlflow.entities.trace_location import TraceLocation
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_TRACKING_URI
from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import SqlIssue
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext


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
                categories=json.dumps(["cost", "COST", " Cost ", "c" * 300]),
                created_timestamp=100,
                last_updated_timestamp=200,
                source_run_id="old-run",
                created_by="author",
            )
        )
    command.upgrade(config, "head")
    store = SqlAlchemyStore(db_url, str(tmp_path / "artifacts"))
    issue = store.get_issue("iss-old")
    assert issue.last_seen_timestamp == 100
    assert issue.last_updated_timestamp == 200
    assert issue.last_updated_by is None
    assert issue.root_causes == ["Old cause"]
    assert issue.categories == ["cost", "COST", " Cost ", "c" * 300]
    assert issue.source_run_id == "old-run"
    assert issue.severity is None
    results = store.search_issues(categories=[" CoSt "], include_total_count=True)
    assert [result.issue_id for result in results] == ["iss-old"]
    assert results.total_count == 1
    assert store.update_issue("iss-old", categories=["latency"]).categories == ["latency"]
    assert store.search_issues(categories=["cost"]) == []


@pytest.mark.notrackingurimock
def test_issue_contract_migration_downgrade_and_reupgrade(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = MLFLOW_TRACKING_URI.get()
    store = WorkspaceAwareSqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))
    config = _get_alembic_config(db_uri)
    issues = {}
    references = {}
    for workspace in ("issue-migration-a", "issue-migration-b"):
        with WorkspaceContext(workspace):
            experiment_id = store.create_experiment(f"issue-round-trip-{uuid.uuid4().hex}")
            run = store.create_run(experiment_id, "author", 1, [], "source")
            issue = store.create_issue(
                experiment_id,
                "Legacy issue",
                "Description",
                source_run_id=run.info.run_id,
                categories=["cost", "custom-category"],
                root_causes=["Legacy cause"],
                metadata={"triage": "investigating"},
                last_seen_timestamp=1234,
                created_by="author",
            )
            store.update_issue(issue.issue_id, last_updated_by="reviewer")
            trace = store.start_trace(
                TraceInfo(
                    trace_id=f"tr-{uuid.uuid4().hex}",
                    trace_location=TraceLocation.from_experiment_id(experiment_id),
                    request_time=1000,
                    execution_duration=100,
                    state=TraceState.OK,
                )
            )
            reference = store.create_assessment(
                IssueReference(
                    issue_id=issue.issue_id, issue_name=issue.name, trace_id=trace.request_id
                )
            )
            issues[workspace] = issue
            references[workspace] = reference

    new_columns = {"metadata", "last_seen_timestamp", "last_updated_by"}
    issue_table = sa.Table("issues", sa.MetaData(), autoload_with=store.engine)
    legacy_columns = [column for column in issue_table.c if column.name not in new_columns]
    legacy_query = sa.select(*legacy_columns).where(
        issue_table.c.issue_id.in_([issue.issue_id for issue in issues.values()])
    )
    with store.engine.connect() as connection:
        original_rows = {
            row.issue_id: dict(row._mapping) for row in connection.execute(legacy_query)
        }
    try:
        command.downgrade(config, "e7d1f4b2a9c6")
        inspector = sa.inspect(store.engine)
        assert "issue_categories" not in inspector.get_table_names()
        assert {column["name"] for column in inspector.get_columns("issues")} == {
            column.name for column in legacy_columns
        }
        assert {index["name"] for index in inspector.get_indexes("issues")} == {
            "index_issues_experiment_id",
            "index_issues_source_run_id",
            "index_issues_status",
        }
        with store.engine.connect() as connection:
            assert {
                row.issue_id: dict(row._mapping) for row in connection.execute(legacy_query)
            } == original_rows
        for workspace, reference in references.items():
            with WorkspaceContext(workspace):
                assert (
                    store.get_assessment(reference.trace_id, reference.assessment_id) == reference
                )
    finally:
        command.upgrade(config, "head")

    for workspace, issue in issues.items():
        with WorkspaceContext(workspace):
            restored = store.get_issue(issue.issue_id)
            assert restored.categories == issue.categories
            assert restored.last_seen_timestamp == issue.created_timestamp
            assert restored.metadata is None
            assert restored.last_updated_by is None
            found = store.search_issues(categories=[" CUSTOM-Category "], include_trace_count=True)
            assert [(result.issue_id, result.trace_count) for result in found] == [
                (issue.issue_id, 1)
            ]
