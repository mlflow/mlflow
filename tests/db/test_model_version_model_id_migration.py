import io
import os
import sqlite3
from importlib import import_module
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations

from mlflow.store.db.utils import _get_alembic_config

MIGRATION = import_module(
    "mlflow.store.db_migrations.versions.4cdafdc631ca_add_logged_model_id_to_model_versions"
)


@pytest.fixture
def migration_table(tmp_path, monkeypatch):
    uri = os.environ["MLFLOW_TRACKING_URI"]
    if uri.startswith("sqlite"):
        uri = f"sqlite:///{tmp_path / 'registry.db'}"
    engine = sa.create_engine(uri)
    # Exercise the migration in the configured database without touching its tracking data.
    table = sa.Table(
        f"model_versions_{uuid4().hex}",
        sa.MetaData(),
        sa.Column("version", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("source", sa.String(500)),
    )
    monkeypatch.setattr(MIGRATION, "_TABLE_NAME", table.name)
    table.create(engine)
    try:
        with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)) as ops:
            yield conn, table, ops
    finally:
        table.drop(engine)
        engine.dispose()


@pytest.mark.notrackingurimock
@pytest.mark.parametrize("resume", [False, True])
def test_model_version_model_id_backfill(migration_table, resume):
    conn, table, ops = migration_table
    direct_id = "m-" + "1" * 32
    legacy_id = "74c31cf8-c82d-48ab-86f5-7a482510a38e"
    sources = [
        (f"models:/{direct_id}", direct_id),
        (f"models:/{legacy_id}", legacy_id),
        (f"models:///{direct_id}", direct_id),
        (f"MODELS://///{legacy_id}", legacy_id),
        ("models://m-authority", None),
        ("models:///m-id/", None),
        ("models:///m-id/extra", None),
        ("models:/m-café", None),
        ("models:/m_tail-01", "m_tail-01"),
        ("models:/m-tail ", None),
        ("models:/registered/1", None),
        ("models:/registered@champion", None),
        ("models://profile@databricks/m-other", None),
        ("models:/", None),
        ("models:/" + "a" * 37, None),
        ("models:/m-123?query", None),
        ("models:/m-123\n", None),
        ("mlflow-artifacts:/1/models/m-123/artifacts", None),
        ("REDACTED-SOURCE-PATH", None),
        (None, None),
    ]
    conn.execute(
        table.insert(),
        [{"version": i, "source": source} for i, (source, _) in enumerate(sources, 1)],
    )
    table.append_column(sa.Column("model_id", sa.String(36)))
    if resume:
        ops.add_column(table.name, sa.Column("model_id", sa.String(36)))
        conn.execute(table.update().where(table.c.version == 1).values(model_id="m-saved"))
        sources[0] = (sources[0][0], "m-saved")

    MIGRATION.upgrade()
    assert (
        conn.execute(sa.select(table.c.source, table.c.model_id).order_by(table.c.version)).all()
        == sources
    )
    MIGRATION.downgrade()
    assert "model_id" not in {col["name"] for col in sa.inspect(conn).get_columns(table.name)}


def test_sqlite_offline_model_id_backfill():
    config = _get_alembic_config("sqlite://")
    config.output_buffer = io.StringIO()
    command.upgrade(config, "f4a9b2c7d1e6:4cdafdc631ca", sql=True)

    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE alembic_version (version_num TEXT)")
        db.execute("CREATE TABLE model_versions (source TEXT)")
        db.executemany(
            "INSERT INTO model_versions VALUES (?)",
            [("models:///m-id",), ("models:/name/2",), ("models:/m-id\x00extra",)],
        )
        db.executescript(config.output_buffer.getvalue())
        assert db.execute("SELECT model_id FROM model_versions ORDER BY rowid").fetchall() == [
            ("m-id",),
            (None,),
            (None,),
        ]
