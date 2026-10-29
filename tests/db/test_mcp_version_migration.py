import importlib
from unittest import mock

import pytest
import sqlalchemy as sa

from mlflow.exceptions import MlflowException


@pytest.fixture
def migration():
    return importlib.import_module(
        "mlflow.store.db_migrations.versions.f4a9b2c7d1e6_make_mcp_versions_case_sensitive"
    )


@pytest.fixture(params=["mysql", "mssql"])
def legacy_database(request, migration):
    # Simulate legacy case-insensitive references without requiring external database servers.
    engine = sa.create_engine("sqlite://")
    collation = "utf8mb4_bin" if request.param == "mysql" else "SQL_Latin1_General_CP1_CS_AS"
    with engine.connect() as connection:
        connection.connection.driver_connection.create_collation(
            collation, lambda left, right: (left > right) - (left < right)
        )
        metadata = sa.MetaData()
        tables = {}
        for table_name, name_column, version_column in (
            ("mcp_server_versions", "name", "version"),
            ("mcp_server_version_tags", "name", "version"),
            ("mcp_server_aliases", "name", "version"),
            ("mcp_access_endpoints", "server_name", "server_version"),
        ):
            tables[table_name] = sa.Table(
                table_name,
                metadata,
                sa.Column("workspace", sa.String(collation="NOCASE")),
                sa.Column(name_column, sa.String(collation="NOCASE")),
                sa.Column(version_column, sa.String(collation="NOCASE")),
            )
        metadata.create_all(connection)
        connection.execute(
            tables["mcp_server_versions"].insert(),
            {"workspace": "default", "name": "server", "version": "1.0.0-RC1"},
        )
        bind = mock.Mock(execute=mock.Mock(wraps=connection.execute))
        bind.dialect.name = request.param
        with (
            mock.patch.object(migration.op, "get_bind", return_value=bind),
            mock.patch.object(migration, "_change_version_type") as change_type,
        ):
            yield connection, tables, bind, change_type
    engine.dispose()


@pytest.mark.parametrize(
    ("table_name", "name_column", "version_column"),
    [
        ("mcp_server_version_tags", "name", "version"),
        ("mcp_server_aliases", "name", "version"),
        ("mcp_access_endpoints", "server_name", "server_version"),
    ],
)
@pytest.mark.parametrize("reference_version", ["1.0.0-rc1", "1.0.0-RC1"])
def test_mcp_migration_checks_references_before_ddl(
    migration, legacy_database, table_name, name_column, version_column, reference_version
):
    connection, tables, bind, change_type = legacy_database
    reference = {"workspace": "default", name_column: "server", version_column: reference_version}
    connection.execute(tables[table_name].insert(), reference)
    if reference_version == "1.0.0-rc1":
        with pytest.raises(MlflowException, match=f"{table_name}.{version_column}") as exc:
            migration.upgrade()
        assert "1.0.0-rc1" in str(exc.value)
        assert "1.0.0-RC1" in str(exc.value)
        assert "workspace 'default'" in str(exc.value)
        assert "before retrying" in str(exc.value)
        change_type.assert_not_called()
    else:
        migration.upgrade()
        change_type.assert_called_once()
    assert all(call.args[0].is_select for call in bind.execute.call_args_list)
    assert connection.execute(sa.select(tables[table_name])).mappings().all() == [reference]
    assert connection.execute(sa.select(tables["mcp_server_versions"].c.version)).scalar() == (
        "1.0.0-RC1"
    )


def test_mcp_migration_allows_alias_endpoints_and_unrelated_versions(migration, legacy_database):
    connection, tables, _, change_type = legacy_database
    connection.execute(
        tables["mcp_access_endpoints"].insert(),
        {"workspace": "default", "server_name": "server", "server_version": None},
    )
    connection.execute(
        tables["mcp_server_versions"].insert(),
        {"workspace": "other", "name": "server", "version": "1.0.0-rc1"},
    )
    connection.execute(
        tables["mcp_server_aliases"].insert(),
        {"workspace": "other", "name": "server", "version": "1.0.0-rc1"},
    )
    migration.upgrade()
    change_type.assert_called_once()


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
def test_mcp_migration_skips_case_sensitive_backends(migration, dialect):
    bind = mock.Mock()
    bind.dialect.name = dialect
    with (
        mock.patch.object(migration.op, "get_bind", return_value=bind),
        mock.patch.object(migration, "_change_version_type") as change_type,
    ):
        migration.upgrade()
    bind.execute.assert_not_called()
    change_type.assert_not_called()
