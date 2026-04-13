from pathlib import Path

import sqlalchemy as sa
from alembic import command

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.utils.workspace_context import WorkspaceContext

REVISION = "dc11669786a5"
PREVIOUS_REVISION = "ae8bbe7743c9"

_LEGACY_JOBS = sa.table(
    "jobs",
    sa.column("id"),
    sa.column("creation_time"),
    sa.column("job_name"),
    sa.column("params"),
    sa.column("workspace"),
    sa.column("timeout"),
    sa.column("status"),
    sa.column("result"),
    sa.column("retry_count"),
    sa.column("last_update_time"),
    sa.column("status_details"),
)


def _prepare_database(tmp_path: Path):
    db_path = tmp_path / "job_persistence_migration.sqlite"
    db_url = f"sqlite:///{db_path}"
    engine = sa.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    config = _get_alembic_config(db_url)
    command.upgrade(config, PREVIOUS_REVISION)
    return engine, config, db_url


def test_job_persistence_migration_adds_schema_and_cancels_legacy_non_terminal_jobs(
    tmp_path: Path,
):
    engine, config, _ = _prepare_database(tmp_path)

    with engine.begin() as conn:
        conn.execute(
            sa.insert(_LEGACY_JOBS),
            [
                {
                    "id": "job-pending",
                    "creation_time": 1000,
                    "job_name": "pending_job",
                    "params": "{}",
                    "workspace": "default",
                    "timeout": None,
                    "status": JobStatus.PENDING.to_int(),
                    "result": None,
                    "retry_count": 0,
                    "last_update_time": 1000,
                    "status_details": None,
                },
                {
                    "id": "job-running",
                    "creation_time": 2000,
                    "job_name": "running_job",
                    "params": "{}",
                    "workspace": "default",
                    "timeout": None,
                    "status": JobStatus.RUNNING.to_int(),
                    "result": None,
                    "retry_count": 0,
                    "last_update_time": 2000,
                    "status_details": None,
                },
                {
                    "id": "job-succeeded",
                    "creation_time": 3000,
                    "job_name": "succeeded_job",
                    "params": "{}",
                    "workspace": "default",
                    "timeout": None,
                    "status": JobStatus.SUCCEEDED.to_int(),
                    "result": "ok",
                    "retry_count": 0,
                    "last_update_time": 3000,
                    "status_details": None,
                },
            ],
        )

    command.upgrade(config, REVISION)

    inspector = sa.inspect(engine)
    assert "job_locks" in inspector.get_table_names()
    assert "scheduler_leases" in inspector.get_table_names()

    job_columns = {column["name"] for column in inspector.get_columns("jobs")}
    assert {
        "executor_backend",
        "lease_expires_at",
        "status_message",
        "progress_payload",
        "progress_updated_at",
        "token_hash",
        "scoped_permissions",
    }.issubset(job_columns)

    jobs = sa.table("jobs", sa.column("id"), sa.column("status"))
    with engine.connect() as conn:
        migrated_statuses = {
            row.id: row.status for row in conn.execute(sa.select(jobs.c.id, jobs.c.status))
        }

    assert migrated_statuses["job-pending"] == JobStatus.CANCELED.to_int()
    assert migrated_statuses["job-running"] == JobStatus.CANCELED.to_int()
    assert migrated_statuses["job-succeeded"] == JobStatus.SUCCEEDED.to_int()


def test_workspace_aware_job_store_works_after_job_persistence_migration(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    _, config, db_url = _prepare_database(tmp_path)
    command.upgrade(config, REVISION)

    store = WorkspaceAwareSqlAlchemyJobStore(db_url)

    with WorkspaceContext("team-a"):
        job = store.create_job("test_job", '{"value": 1}')
        fetched = store.get_job(job.job_id)

    assert fetched.workspace == "team-a"
    assert fetched.status == JobStatus.PENDING
