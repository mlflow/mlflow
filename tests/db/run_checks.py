import os
import argparse

import sqlalchemy
from sqlalchemy.schema import MetaData, CreateTable

import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

assert _TRACKING_URI_ENV_VAR in os.environ


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
        print("Tracking URI:", mlflow.get_tracking_uri())
        mlflow.log_param("p", "param")
        mlflow.log_metric("m", 1.0)
        mlflow.set_tag("t", "tag")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MockModel(),
            registered_model_name="mock",
        )
        print(mlflow.get_run(run.info.run_id))


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


def main():
    args = parse_args()
    schema = get_db_schema()
    title = "Schema"
    print("=" * 10, title, "=", 10)
    print(schema)
    print("=" * (20 + 2 + len(title)))
    os.makedirs(os.path.dirname(args.schema_output), exist_ok=True)
    with open(args.schema_output, "w") as f:
        f.write(schema)

    run_logging_operations()


if __name__ == "__main__":
    main()
