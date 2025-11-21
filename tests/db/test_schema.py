import difflib
import logging
import re
from pathlib import Path
from typing import NamedTuple

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.schema import CreateTable, MetaData, UniqueConstraint

_logger = logging.getLogger(__name__)

_DIALECT_REFLECTED_UNIQUE_CONSTRAINTS = {
    "mysql": {"uq_experiments_workspace_name"},
    "mssql": {"uq_experiments_workspace_name"},
}

import mlflow
from mlflow.environment_variables import MLFLOW_TRACKING_URI

pytestmark = pytest.mark.notrackingurimock


def get_database_dialect(uri):
    return create_engine(uri).dialect.name


def get_tracking_uri():
    return MLFLOW_TRACKING_URI.get()


def dump_schema(db_uri):
    engine = create_engine(db_uri)
    created_tables_metadata = MetaData()
    created_tables_metadata.reflect(bind=engine)
    _reattach_missing_unique_constraints(engine, created_tables_metadata)
    # Write out table schema as described in
    # https://docs.sqlalchemy.org/en/13/faq/metadata_schema.html#how-can-i-get-the-create-table-drop-table-output-as-a-string
    lines = []
    for table in created_tables_metadata.sorted_tables:
        # Apply `str.rstrip` to remove trailing whitespaces
        lines += map(str.rstrip, str(CreateTable(table)).splitlines())
    return "\n".join(lines)


def _reattach_missing_unique_constraints(engine, metadata):
    constraint_names = _DIALECT_REFLECTED_UNIQUE_CONSTRAINTS.get(engine.dialect.name)
    if not constraint_names:
        return
    inspector = inspect(engine)
    for table in metadata.sorted_tables:
        existing_unique_columns = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        # Not all dialects reflect `UniqueConstraint` objects the same way. MySQL reports
        # them as indexes; MSSQL doesn't implement `get_unique_constraints` at all. We
        # normalize the reflection results via `_get_unique_constraints` so the same code
        # path can reattach missing `UniqueConstraint`s across dialects.
        for unique in _get_unique_constraints(inspector, engine.dialect.name, table.name):
            name = unique.get("name")
            columns = tuple(unique.get("column_names") or ())
            duplicates_index = unique.get("duplicates_index")
            if not columns or name not in constraint_names:
                continue
            if engine.dialect.name == "mysql" and not duplicates_index:
                # MySQL exposes unique constraints as unique indexes. SQLAlchemy treats those as
                # indexes during reflection, so the reflected metadata lacks the original
                # `UniqueConstraint`. Only recreate constraints that are backed by an actual
                # unique index reported via `duplicates_index`.
                continue
            if columns in existing_unique_columns:
                continue
            missing_columns = tuple(column for column in columns if column not in table.c)
            if missing_columns:
                _logger.warning(
                    "Skipping recreation of unique constraint '%s' on table '%s' due to "
                    "missing columns: %s",
                    name,
                    table.name,
                    ", ".join(missing_columns),
                )
                continue
            constraint = UniqueConstraint(*[table.c[column] for column in columns], name=name)
            table.append_constraint(constraint)
            existing_unique_columns.add(columns)


def _get_unique_constraints(inspector, dialect, table_name):
    try:
        unique_constraints = inspector.get_unique_constraints(table_name)
    except NotImplementedError:
        unique_constraints = None
    if unique_constraints is None:
        unique_constraints = []
    if not unique_constraints:
        unique_constraints = [
            {
                "name": index.get("name"),
                "column_names": index.get("column_names"),
                "duplicates_index": index.get("unique"),
            }
            for index in inspector.get_indexes(table_name)
            if index.get("unique")
        ]
    return unique_constraints


class _CreateTable(NamedTuple):
    table: str
    columns: str


_CREATE_TABLE_REGEX = re.compile(
    r"""
CREATE TABLE (?P<table>\S+?) \(
(?P<columns>.+?)
\)
""".strip(),
    flags=re.DOTALL,
)


def parse_create_tables(schema):
    return [
        _CreateTable(
            table=m.group("table"),
            columns=set(m.group("columns").splitlines()),
        )
        for m in _CREATE_TABLE_REGEX.finditer(schema)
    ]


def schema_equal(schema_a, schema_b):
    create_tables_a = parse_create_tables(schema_a)
    create_tables_b = parse_create_tables(schema_b)
    assert create_tables_a != []
    assert create_tables_b != []
    return create_tables_a == create_tables_b


def get_schema_path(db_uri):
    return Path(__file__).parent / "schemas" / (get_database_dialect(db_uri) + ".sql")


def iter_parameter_sets():
    a = """
CREATE TABLE table (
    col VARCHAR(10)
)
"""
    b = """
CREATE TABLE table (
    col VARCHAR(10)
)
"""
    yield pytest.param(a, b, True, id="identical schemas")

    a = """
CREATE TABLE table1 (
    col VARCHAR(10)
)
"""
    b = """
CREATE TABLE table2 (
    col VARCHAR(10)
)
"""
    yield pytest.param(a, b, False, id="different table names")

    a = """
CREATE TABLE table (
    col1 VARCHAR(10)
)
"""
    b = """
CREATE TABLE table (
    col2 VARCHAR(10)
)
"""
    yield pytest.param(a, b, False, id="different column names")


@pytest.mark.parametrize(("a", "b", "expected"), iter_parameter_sets())
def test_schema_equal(a, b, expected):
    assert schema_equal(a, b) is expected


def initialize_database():
    with mlflow.start_run():
        pass


def get_schema_update_command(dialect):
    this_script = Path(__file__).relative_to(Path.cwd())
    docker_compose_yml = this_script.parent / "compose.yml"
    return f"docker compose -f {docker_compose_yml} run --rm mlflow-{dialect} python {this_script}"


def test_schema_is_up_to_date():
    initialize_database()
    tracking_uri = get_tracking_uri()
    schema_path = get_schema_path(tracking_uri)
    existing_schema = schema_path.read_text()
    latest_schema = dump_schema(tracking_uri)
    dialect = get_database_dialect(tracking_uri)
    update_command = get_schema_update_command(dialect)
    message = (
        f"{schema_path.relative_to(Path.cwd())} is not up-to-date. "
        f"Please run this command to update it: {update_command}"
    )
    diff = "".join(
        difflib.ndiff(
            existing_schema.splitlines(keepends=True), latest_schema.splitlines(keepends=True)
        )
    )
    rel_path = schema_path.relative_to(Path.cwd())
    message = f"""
=================================== EXPECTED ===================================
{latest_schema}
==================================== ACTUAL ====================================
{existing_schema}
===================================== DIFF =====================================
{diff}
================================== HOW TO FIX ==================================
Manually copy & paste the expected schema in {rel_path} or run the following command:
{update_command}
"""
    assert schema_equal(existing_schema, latest_schema), message


def main():
    tracking_uri = get_tracking_uri()
    assert tracking_uri, f"Environment variable {MLFLOW_TRACKING_URI} must be set"
    get_database_dialect(tracking_uri)  # Ensure `tracking_uri` is a database URI
    mlflow.set_tracking_uri(tracking_uri)
    initialize_database()
    schema_path = get_schema_path(tracking_uri)
    existing_schema = schema_path.read_text()
    latest_schema = dump_schema(tracking_uri)
    if not schema_equal(existing_schema, latest_schema):
        schema_path.write_text(latest_schema)


if __name__ == "__main__":
    main()
