import os
import re
import argparse

import sqlalchemy
from sqlalchemy.schema import MetaData, CreateTable

import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR


class MockModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-output", required=True, help="Output path of DB schema")
    return parser.parse_args()


def run_logging_operations():
    with mlflow.start_run() as run:
        mlflow.log_param("p", "param")
        mlflow.log_metric("m", 1.0)
        mlflow.set_tag("t", "tag")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MockModel(),
            registered_model_name="mock",
        )

    # Ensure the following migration scripts work correctly:
    # - cfd24bdc0731_update_run_status_constraint_with_killed.py
    # - 0a8213491aaa_drop_duplicate_killed_constraint.py
    client = mlflow.tracking.MlflowClient()
    client.set_terminated(run_id=run.info.run_id, status="KILLED")


def get_db_schema():
    engine = sqlalchemy.create_engine(mlflow.get_tracking_uri())
    created_tables_metadata = MetaData(bind=engine)
    created_tables_metadata.reflect()
    # Write out table schema as described in
    # https://docs.sqlalchemy.org/en/13/faq/metadata_schema.html#how-can-i-get-the-create-table-drop-table-output-as-a-string
    lines = []
    for ti in created_tables_metadata.sorted_tables:
        lines += list(map(str.rstrip, str(CreateTable(ti)).splitlines()))
    return "\n".join(lines)


def get_create_tables(schema):
    pattern = r"""
CREATE TABLE (?P<table_name>\S+?) \(
(?P<columns_and_constraints>\S+?)
\)
""".strip()
    return list(re.finditer(pattern, schema, flags=re.DOTALL))


def is_schema_changed(new, old):
    tables_new = get_create_tables(new)
    tables_old = get_create_tables(old)

    if len(tables_new) != len(tables_old):
        return False

    for table_new, table_old in zip(tables_new, tables_old):
        if table_new.group("table_name") != table_old.group("table_name"):
            return False

        cols_new = table_new.group("columns_and_constraints").splitlines()
        cols_old = table_old.group("columns_and_constraints").splitlines()
        # Compare as a set to ignore the ordering of columns and constraints
        if set(cols_new) != set(cols_old):
            False

    return True


def write_file(s, path):
    with open(path, "w") as f:
        f.write(s)


def main():
    assert _TRACKING_URI_ENV_VAR in os.environ
    print("Tracking URI:", os.environ.get(_TRACKING_URI_ENV_VAR))

    args = parse_args()
    run_logging_operations()
    schema = get_db_schema()
    schema_output = args.schema_output
    os.makedirs(os.path.dirname(schema_output), exist_ok=True)
    if os.path.exists(schema_output):
        with open(schema_output) as f:
            existing_schema = f.read()
        if not is_schema_changed(schema, existing_schema):
            write_file(schema, schema_output)
    else:
        write_file(schema, schema_output)


if __name__ == "__main__":
    main()
