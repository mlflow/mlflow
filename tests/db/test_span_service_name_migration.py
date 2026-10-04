import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase

REVISION = "e8f9a0b1c2d3"
PREVIOUS_REVISION = "dc11669786a5"

pytestmark = pytest.mark.notrackingurimock


def test_span_service_name_migration(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'span_service_name_migration.sqlite'}"
    engine = sa.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    config = _get_alembic_config(db_url)
    command.upgrade(config, PREVIOUS_REVISION)

    def span_columns():
        engine.dispose()
        return {column["name"] for column in sa.inspect(engine).get_columns("spans")}

    assert "service_name" not in span_columns()

    with engine.begin() as conn:
        metadata = sa.MetaData()
        experiments = sa.Table("experiments", metadata, autoload_with=conn)
        trace_info = sa.Table("trace_info", metadata, autoload_with=conn)
        trace_tags = sa.Table("trace_tags", metadata, autoload_with=conn)
        spans = sa.Table("spans", metadata, autoload_with=conn)
        conn.execute(
            experiments.insert(),
            {
                "experiment_id": 1,
                "name": "service-name-migration",
                "artifact_location": "file:///tmp/artifacts",
                "lifecycle_stage": "active",
                "creation_time": 1,
                "last_update_time": 1,
            },
        )
        trace_ids = ["trace-tag", "trace-attribute", "trace-missing"]
        conn.execute(
            trace_info.insert(),
            [
                {
                    "request_id": trace_id,
                    "experiment_id": 1,
                    "timestamp_ms": 1,
                    "execution_time_ms": 1,
                    "status": "OK",
                }
                for trace_id in trace_ids
            ],
        )
        conn.execute(
            trace_tags.insert(),
            {"request_id": "trace-tag", "key": "service.name", "value": "tag-service"},
        )

        def span_row(trace_id, span_id, content):
            return {
                "trace_id": trace_id,
                "experiment_id": 1,
                "span_id": span_id,
                "name": span_id,
                "type": "CHAIN",
                "status": "OK",
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
                "content": content,
            }

        conn.execute(
            spans.insert(),
            [
                span_row("trace-tag", "tag-only", "{}"),
                span_row(
                    "trace-tag",
                    "span-attribute-wins",
                    json.dumps({"attributes": {"service.name": json.dumps("span-service")}}),
                ),
                span_row(
                    "trace-attribute",
                    "attribute-only",
                    json.dumps({"attributes": {"service.name": json.dumps("attribute-service")}}),
                ),
                span_row("trace-missing", "missing", "{}"),
            ],
        )

    command.upgrade(config, REVISION)
    assert "service_name" in span_columns()
    with engine.connect() as conn:
        spans = sa.Table("spans", sa.MetaData(), autoload_with=conn)
        assert conn.execute(
            sa.select(spans.c.span_id, spans.c.service_name).order_by(spans.c.span_id)
        ).all() == [
            ("attribute-only", "attribute-service"),
            ("missing", None),
            ("span-attribute-wins", "span-service"),
            ("tag-only", None),
        ]

    command.downgrade(config, PREVIOUS_REVISION)
    assert "service_name" not in span_columns()
