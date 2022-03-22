import os
import re
from pathlib import Path
from collections import namedtuple

import pytest
from sqlalchemy.schema import MetaData, CreateTable
from sqlalchemy import create_engine

import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR


pytestmark = pytest.mark.notrackingurimock


def get_database_dialect(uri):
    return create_engine(uri).dialect.name


def get_tracking_uri():
    return os.getenv(_TRACKING_URI_ENV_VAR)


def dump_schema(db_uri):
    engine = create_engine(db_uri)
    created_tables_metadata = MetaData(bind=engine)
    created_tables_metadata.reflect()
    # Write out table schema as described in
    # https://docs.sqlalchemy.org/en/13/faq/metadata_schema.html#how-can-i-get-the-create-table-drop-table-output-as-a-string
    lines = []
    for table in created_tables_metadata.sorted_tables:
        # Apply `str.rstrip` to remove trailing whitespaces
        lines += map(str.rstrip, str(CreateTable(table)).splitlines())
    return "\n".join(lines)


_CreateTable = namedtuple("_CreateTable", ["table", "columns"])


_CREATE_TABLE_REGEX = re.compile(
    r"""
CREATE TABLE (?P<table>\S+?) \(
(?P<columns>.+?)
\)
""".strip(),
    flags=re.DOTALL,
)


def parse_create_table_statements(schema):
    return [
        _CreateTable(
            table=m.group("table"),
            columns=set(m.group("columns").splitlines()),
        )
        for m in _CREATE_TABLE_REGEX.finditer(schema)
    ]


def schema_equal(schema_a, schema_b):
    create_tables_a = parse_create_table_statements(schema_a)
    create_tables_b = parse_create_table_statements(schema_b)
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


@pytest.mark.parametrize("a, b, expected", iter_parameter_sets())
def test_schema_equal(a, b, expected):
    assert schema_equal(a, b) is expected


def initialize_database():
    with mlflow.start_run():
        pass


def test_schema_is_up_to_date():
    initialize_database()
    tracking_uri = get_tracking_uri()
    schema_path = get_schema_path(tracking_uri)
    existing_schema = schema_path.read_text()
    latest_schema = dump_schema(tracking_uri)
    this_script_rel = Path(__file__).relative_to(Path.cwd())
    docker_compose_yml = this_script_rel.parent / "docker-compose.yml"
    dialect = get_database_dialect(tracking_uri)
    schema_path_rel = schema_path.relative_to(Path.cwd())
    update_command = (
        f"docker-compose -f {docker_compose_yml} run --rm mlflow-{dialect} python {this_script_rel}"
    )
    message = (
        f"{schema_path_rel} is not up-to-date. "
        f"Please run the this command to update it: {update_command}"
    )
    assert schema_equal(existing_schema, latest_schema), message


def main():
    tracking_uri = get_tracking_uri()
    assert tracking_uri, f"Environment variable {_TRACKING_URI_ENV_VAR} must be set"
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
