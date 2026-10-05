import json
import logging
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.db_migrations.versions import (
    e8f9a0b1c2d3_add_service_name_to_spans as service_name_migration,
)
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase

REVISION = "e8f9a0b1c2d3"
PREVIOUS_REVISION = "dc11669786a5"

pytestmark = pytest.mark.notrackingurimock


@pytest.mark.parametrize(
    ("supports_sane_rowcount", "rowcount", "warning"),
    [
        (True, 2, None),
        (
            True,
            1,
            "Span service-name backfill updated 1 of 2 rows; unmatched rows will keep a NULL "
            "service_name",
        ),
        (False, -1, None),
    ],
)
def test_execute_backfill_updates_warns_on_supported_rowcount_mismatch(
    caplog, supports_sane_rowcount, rowcount, warning
):
    result = mock.Mock(rowcount=rowcount)
    result.supports_sane_multi_rowcount.return_value = supports_sane_rowcount
    bind = mock.Mock()
    bind.execute.return_value = result
    updates = [{"id": 1}, {"id": 2}]

    with caplog.at_level(logging.WARNING, logger=service_name_migration.__name__):
        service_name_migration._execute_backfill_updates(bind, "update", updates)

    bind.execute.assert_called_once_with("update", updates)
    assert [record.getMessage() for record in caplog.records] == ([warning] if warning else [])


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

    with engine.begin() as conn:
        spans = sa.Table("spans", sa.MetaData(), autoload_with=conn)
        conn.execute(
            spans
            .update()
            .where(spans.c.span_id == "attribute-only")
            .values(
                content=json.dumps({
                    "attributes": {"service.name": json.dumps("replacement-service")}
                }),
                service_name="preserved-service",
            )
        )
        conn.execute(
            spans.update().where(spans.c.span_id == "span-attribute-wins").values(service_name=None)
        )
        with mock.patch.object(service_name_migration.op, "get_bind", return_value=conn):
            service_name_migration._backfill_service_names()

        assert conn.execute(
            sa
            .select(spans.c.span_id, spans.c.service_name)
            .where(spans.c.span_id.in_(["attribute-only", "span-attribute-wins"]))
            .order_by(spans.c.span_id)
        ).all() == [
            ("attribute-only", "preserved-service"),
            ("span-attribute-wins", "span-service"),
        ]

    command.downgrade(config, PREVIOUS_REVISION)
    assert "service_name" not in span_columns()
