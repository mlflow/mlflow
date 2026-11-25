import io
import os
import sqlite3

import pytest
import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase


@pytest.fixture
def migration_db(tmp_path):
    uri = os.environ["MLFLOW_TRACKING_URI"]
    external = not uri.startswith("sqlite")
    if not external:
        uri = f"sqlite:///{tmp_path / 'registry.db'}"
    engine = sa.create_engine(uri)
    config = _get_alembic_config(uri)
    try:
        with engine.begin() as conn:
            if external:
                metadata = sa.MetaData()
                metadata.reflect(conn)
                metadata.drop_all(conn)
            InitialBase.metadata.create_all(conn)
        command.upgrade(config, "f4a9b2c7d1e6")
        yield engine, config
    finally:
        if external:
            command.upgrade(config, "head")
        engine.dispose()


@pytest.mark.notrackingurimock
@pytest.mark.parametrize("resume", [False, True])
def test_model_version_model_id_backfill(migration_db, resume):
    engine, config = migration_db
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
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO registered_models (name) VALUES ('model')"))
        conn.execute(
            sa.text("INSERT INTO model_versions (name, version, source) VALUES ('model', :v, :s)"),
            [{"v": i, "s": source} for i, (source, _) in enumerate(sources, 1)],
        )

        if resume:
            conn.execute(sa.text("ALTER TABLE model_versions ADD model_id VARCHAR(36)"))
            conn.execute(
                sa.text("UPDATE model_versions SET model_id = 'm-saved' WHERE version = 1")
            )
            sources[0] = (sources[0][0], "m-saved")

    command.upgrade(config, "4cdafdc631ca")
    with engine.connect() as conn:
        assert (
            conn.execute(
                sa.text("SELECT source, model_id FROM model_versions ORDER BY version")
            ).all()
            == sources
        )
    command.downgrade(config, "f4a9b2c7d1e6")
    assert "model_id" not in {
        col["name"] for col in sa.inspect(engine).get_columns("model_versions")
    }


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
