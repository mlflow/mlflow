import os
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase

pytestmark = pytest.mark.notrackingurimock

DB_URI = os.environ.get("MLFLOW_TRACKING_URI")
USE_EXTERNAL_DB = DB_URI is not None and not DB_URI.startswith("sqlite")

MIGRATION_MODULE = import_module(
    "mlflow.store.db_migrations.versions.3ecac0fa87a0_backfill_traced_experiment_kind"
)
REVISION = MIGRATION_MODULE.revision
PREVIOUS_REVISION = MIGRATION_MODULE.down_revision

KIND = "mlflow.experimentKind"

_EXPERIMENTS = sa.table(
    "experiments",
    sa.column("experiment_id"),
    sa.column("name"),
    sa.column("workspace"),
    sa.column("lifecycle_stage"),
)
_EXPERIMENT_TAGS = sa.table(
    "experiment_tags", sa.column("key"), sa.column("value"), sa.column("experiment_id")
)
_TRACE_INFO = sa.table(
    "trace_info",
    sa.column("request_id"),
    sa.column("experiment_id"),
    sa.column("timestamp_ms"),
    sa.column("status"),
)


@pytest.fixture(scope="module", autouse=True)
def _upgrade_external_db_to_head_after_module():
    yield
    if USE_EXTERNAL_DB:
        command.upgrade(_get_alembic_config(DB_URI), "head")


@contextmanager
def _identity_insert(conn, table_name):
    if conn.dialect.name != "mssql":
        yield
        return

    conn.execute(sa.text(f"SET IDENTITY_INSERT {table_name} ON"))
    try:
        yield
    finally:
        conn.execute(sa.text(f"SET IDENTITY_INSERT {table_name} OFF"))


def _prepare_database(tmp_path: Path):
    if USE_EXTERNAL_DB:
        engine = sa.create_engine(DB_URI)
        with engine.begin() as conn:
            metadata = sa.MetaData()
            metadata.reflect(bind=conn)
            metadata.drop_all(bind=conn)
            InitialBase.metadata.create_all(conn)
        config = _get_alembic_config(DB_URI)
    else:
        db_url = f"sqlite:///{tmp_path / 'traced_experiment_kind_migration.sqlite'}"
        engine = sa.create_engine(db_url)
        InitialBase.metadata.create_all(engine)
        config = _get_alembic_config(db_url)
    command.upgrade(config, PREVIOUS_REVISION)
    return engine, config


def _insert_experiment(conn, experiment_id, workspace, has_traces, kind=None):
    with _identity_insert(conn, "experiments"):
        conn.execute(
            sa.insert(_EXPERIMENTS).values(
                experiment_id=experiment_id,
                name=f"experiment-{experiment_id}",
                workspace=workspace,
                lifecycle_stage="active",
            )
        )
    if kind:
        conn.execute(
            sa.insert(_EXPERIMENT_TAGS).values(key=KIND, value=kind, experiment_id=experiment_id)
        )
    if has_traces:
        # Two traces, to check the experiment is tagged once
        for n in range(2):
            conn.execute(
                sa.insert(_TRACE_INFO).values(
                    request_id=f"tr-{experiment_id}-{n}",
                    experiment_id=experiment_id,
                    timestamp_ms=0,
                    status="OK",
                )
            )


def _kinds(engine) -> dict[int, str]:
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(_EXPERIMENT_TAGS.c.experiment_id, _EXPERIMENT_TAGS.c.value).where(
                _EXPERIMENT_TAGS.c.key == KIND
            )
        )
        return dict(rows.all())


def test_migration_backfills_kind_of_traced_experiments(tmp_path: Path):
    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _insert_experiment(conn, 101, "team-a", has_traces=True)
            _insert_experiment(conn, 102, "team-b", has_traces=True)
            _insert_experiment(
                conn, 103, "team-a", has_traces=True, kind="custom_model_development"
            )
            _insert_experiment(conn, 104, "team-a", has_traces=False)

        command.upgrade(config, REVISION)

        assert _kinds(engine) == {
            101: "genai_development_inferred",
            102: "genai_development_inferred",
            103: "custom_model_development",
        }
    finally:
        engine.dispose()


def test_migration_downgrade_keeps_backfilled_kinds(tmp_path: Path):
    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _insert_experiment(conn, 101, "default", has_traces=True)
        command.upgrade(config, REVISION)

        command.downgrade(config, PREVIOUS_REVISION)
        assert _kinds(engine) == {101: "genai_development_inferred"}

        # Re-upgrading must not fail on the tag the first upgrade inserted
        command.upgrade(config, REVISION)
        assert _kinds(engine) == {101: "genai_development_inferred"}
    finally:
        engine.dispose()
