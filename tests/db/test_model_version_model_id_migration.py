import pytest
import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase


@pytest.mark.parametrize("resume", [False, True])
def test_model_version_model_id_backfill(tmp_path, resume):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    config = _get_alembic_config(uri)
    engine = sa.create_engine(uri)
    InitialBase.metadata.create_all(engine)
    command.upgrade(config, "f4a9b2c7d1e6")

    direct_id = "m-" + "1" * 32
    legacy_id = "74c31cf8-c82d-48ab-86f5-7a482510a38e"
    sources = [
        (f"models:/{direct_id}", direct_id),
        (f"models:/{legacy_id}", legacy_id),
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
            conn.execute(sa.text("ALTER TABLE model_versions ADD COLUMN model_id VARCHAR(36)"))
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
