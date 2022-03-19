import os
import re
import tempfile
from pathlib import Path
from collections import namedtuple

import pytest
import sqlalchemy
from sqlalchemy.schema import MetaData, CreateTable
from sqlalchemy import create_engine

import mlflow
import mlflow.db
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR
from mlflow.entities import ViewType


pytestmark = pytest.mark.notrackingurimock


def get_database_dialect(uri):
    return create_engine(uri).dialect.name


def get_tracking_uri():
    return os.getenv(_TRACKING_URI_ENV_VAR)


def as_sqlite_uri(path):
    return "sqlite:///" + str(path)


@pytest.fixture(autouse=True)
def mock_tracking_uri_env_var(tmp_path, monkeypatch):
    if _TRACKING_URI_ENV_VAR not in os.environ:
        monkeypatch.setenv(_TRACKING_URI_ENV_VAR, as_sqlite_uri(tmp_path / "mlruns.sqlite"))


def dump_schema(db_uri):
    engine = sqlalchemy.create_engine(db_uri)
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


class Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


def start_run_and_log_data():
    with mlflow.start_run():
        mlflow.log_param("p", "param")
        mlflow.log_metric("m", 1.0)
        mlflow.set_tag("t", "tag")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=Model(),
            registered_model_name="model",
        )


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


def test_search_runs():
    start_run_and_log_data()
    runs = mlflow.search_runs(experiment_ids=["0"], order_by=["param.start_time DESC"])
    mlflow.get_run(runs["run_id"][0])


def test_list_experiments():
    start_run_and_log_data()
    experiments = mlflow.list_experiments(view_type=ViewType.ALL, max_results=5)
    assert len(experiments) > 0


def test_set_run_status_to_killed():
    """
    This test ensures the following migration scripts work correctly:
    - cfd24bdc0731_update_run_status_constraint_with_killed.py
    - 0a8213491aaa_drop_duplicate_killed_constraint.py
    """
    with mlflow.start_run() as run:
        pass
    client = mlflow.tracking.MlflowClient()
    client.set_terminated(run_id=run.info.run_id, status="KILLED")


def test_schema_is_ud_to_date():
    start_run_and_log_data()
    tracking_uri = get_tracking_uri()
    schema_path = get_schema_path(tracking_uri)
    existing_schema = schema_path.read_text()
    latest_schema = dump_schema(tracking_uri)
    message = (
        "{schema_path} is not up-to-date. Please run `python {this_script}` to update it.".format(
            schema_path=schema_path, this_script=Path(__file__).relative_to(Path.cwd())
        )
    )
    assert schema_equal(existing_schema, latest_schema), message


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a temporary sqlite database if "MLFLOW_TRACKING_URI" is not set
        tracking_uri = get_tracking_uri() or as_sqlite_uri(Path(tmpdir) / "mlruns.sqlite")
        # Make sure `tracking_uri` is a database URI
        get_database_dialect(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        start_run_and_log_data()
        schema_path = get_schema_path(tracking_uri)
        existing_schema = schema_path.read_text()
        latest_schema = dump_schema(tracking_uri)
        if not schema_equal(existing_schema, latest_schema):
            schema_path.write_text(latest_schema)


if __name__ == "__main__":
    main()
