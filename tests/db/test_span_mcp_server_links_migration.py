from importlib import import_module

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite
from sqlalchemy.orm import Session

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import SqlSpanMCPServerLink, SqlTraceInfo
from mlflow.utils.time import get_current_time_millis

MIGRATION = import_module(
    "mlflow.store.db_migrations.versions.c8f3a1b9d2e7_add_span_mcp_server_links"
)


@pytest.fixture(params=["migration", "orm"])
def engine(db_uri, request):
    engine = sa.create_engine(db_uri)
    with engine.begin() as conn:
        conn.execute(sa.text("PRAGMA foreign_keys = ON"))
        if request.param == "orm":
            SqlSpanMCPServerLink.__table__.drop(conn)
            SqlSpanMCPServerLink.__table__.create(conn)
        conn.execute(
            SqlTraceInfo.__table__.insert(),
            [
                {"request_id": trace_id, "experiment_id": 0, "timestamp_ms": 1, "status": "OK"}
                for trace_id in ("trace-1", "trace-2")
            ],
        )
    try:
        yield engine
    finally:
        engine.dispose()


def _link(**overrides):
    return {
        "trace_id": "trace-1",
        "span_id": "span-1",
        "mcp_workspace": "team-a",
        "mcp_name": "io.acme/weather",
        "mcp_version": None,
        "created_at": 1_700_000_000_000,
        **overrides,
    }


def test_schema(engine):
    inspector = sa.inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("span_mcp_server_links")}
    assert set(columns) == set(_link())
    for name, length in (
        ("trace_id", 50),
        ("span_id", 50),
        ("mcp_workspace", 63),
        ("mcp_name", 256),
        ("mcp_version", 256),
    ):
        assert columns[name]["type"].length == length
    assert isinstance(columns["created_at"]["type"], sa.BigInteger)
    assert {name for name, column in columns.items() if column["nullable"]} == {"mcp_version"}
    assert all(column["default"] is None for column in columns.values())
    assert inspector.get_pk_constraint("span_mcp_server_links")["constrained_columns"] == [
        "trace_id",
        "span_id",
        "mcp_workspace",
        "mcp_name",
    ]
    assert inspector.get_indexes("span_mcp_server_links") == [
        {
            "name": "ix_span_mcp_server_links_version",
            "column_names": ["mcp_workspace", "mcp_name", "mcp_version", "trace_id"],
            "unique": 0,
            "dialect_options": {},
        }
    ]
    (foreign_key,) = inspector.get_foreign_keys("span_mcp_server_links")
    assert foreign_key["constrained_columns"] == ["trace_id"]
    assert foreign_key["referred_table"] == "trace_info"
    assert foreign_key["referred_columns"] == ["request_id"]
    assert foreign_key["options"]["ondelete"] == "CASCADE"


def test_observations_without_span_or_registry_entry(engine):
    table = SqlSpanMCPServerLink.__table__
    rows = [
        _link(),
        _link(trace_id="trace-2"),
        _link(span_id="span-2", mcp_version="1.0.0"),
        _link(mcp_workspace="team-b"),
        _link(mcp_name="io.acme/search"),
    ]
    with engine.begin() as conn:
        conn.execute(table.insert(), rows)
        actual = conn.execute(sa.select(table)).mappings().all()
        assert len(actual) == len(rows)
        assert all(row in actual for row in rows)


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_primary_key_excludes_version(engine, version):
    with engine.begin() as conn:
        conn.execute(SqlSpanMCPServerLink.__table__.insert(), _link())
        with pytest.raises(sa.exc.IntegrityError, match="UNIQUE constraint failed"):
            conn.execute(SqlSpanMCPServerLink.__table__.insert(), _link(mcp_version=version))


def test_missing_trace_is_rejected(engine):
    with engine.begin() as conn:
        with pytest.raises(sa.exc.IntegrityError, match="FOREIGN KEY"):
            conn.execute(SqlSpanMCPServerLink.__table__.insert(), _link(trace_id="missing"))


@pytest.mark.parametrize("orm_delete", [False, True])
def test_trace_deletion_cascades(engine, orm_delete):
    with Session(engine) as session:
        # ORM deletion must also clean up links when SQLite FK enforcement is disabled.
        session.execute(sa.text(f"PRAGMA foreign_keys = {'OFF' if orm_delete else 'ON'}"))
        session.add_all([
            SqlSpanMCPServerLink(**_link()),
            SqlSpanMCPServerLink(**_link(span_id="span-2", mcp_version="1.0.0")),
            SqlSpanMCPServerLink(**_link(trace_id="trace-2")),
        ])
        session.commit()
        if orm_delete:
            session.delete(session.get(SqlTraceInfo, "trace-1"))
        else:
            session.execute(sa.delete(SqlTraceInfo).where(SqlTraceInfo.request_id == "trace-1"))
        session.commit()
        assert session.scalars(sa.select(SqlSpanMCPServerLink.trace_id)).all() == ["trace-2"]


def test_created_at_uses_application_timestamp(engine):
    with Session(engine) as session:
        values = _link()
        del values["created_at"]
        link = SqlSpanMCPServerLink(**values)
        before = get_current_time_millis()
        session.add(link)
        session.flush()
        assert before <= link.created_at <= get_current_time_millis()


@pytest.mark.parametrize("dialect", [sqlite, postgresql, mysql, mssql])
def test_migration_ddl_matches_orm(monkeypatch, dialect):
    statements = []
    engine = sa.create_mock_engine(
        f"{dialect.dialect.name}://",
        lambda statement, *args, **kwargs: statements.append(statement),
    )
    monkeypatch.setattr(MIGRATION, "op", Operations(MigrationContext.configure(engine)))
    MIGRATION.upgrade()
    create_table, create_index = statements
    assert str(create_table.compile(dialect=dialect.dialect())) == str(
        sa.schema.CreateTable(SqlSpanMCPServerLink.__table__).compile(dialect=dialect.dialect())
    )
    assert str(create_index.compile(dialect=dialect.dialect())) == str(
        sa.schema.CreateIndex(next(iter(SqlSpanMCPServerLink.__table__.indexes))).compile(
            dialect=dialect.dialect()
        )
    )


def test_upgrade_and_downgrade_preserve_traces(tmp_path):
    uri = f"sqlite:///{tmp_path / 'migration.db'}"
    engine = sa.create_engine(uri)
    try:
        InitialBase.metadata.create_all(engine)
        config = _get_alembic_config(uri)
        command.upgrade(config, MIGRATION.down_revision)
        with engine.begin() as conn:
            conn.execute(
                sa.text("INSERT INTO experiments (experiment_id, name) VALUES (1, 'legacy')")
            )
            conn.execute(
                SqlTraceInfo.__table__.insert(),
                {"request_id": "legacy", "experiment_id": 1, "timestamp_ms": 1, "status": "OK"},
            )
        command.upgrade(config, MIGRATION.revision)
        with engine.begin() as conn:
            conn.execute(SqlSpanMCPServerLink.__table__.insert(), _link(trace_id="legacy"))
        command.downgrade(config, MIGRATION.down_revision)
        assert "span_mcp_server_links" not in sa.inspect(engine).get_table_names()
        with engine.connect() as conn:
            assert conn.execute(sa.select(SqlTraceInfo.request_id)).scalars().all() == ["legacy"]
        command.upgrade(config, MIGRATION.revision)
        with engine.connect() as conn:
            assert conn.execute(sa.select(SqlSpanMCPServerLink.__table__)).all() == []
    finally:
        engine.dispose()
