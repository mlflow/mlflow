"""
Usage
-----
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# pre migration
python tests/db/check_migration.py pre-migration

# post migration
python tests/db/check_migration.py post-migration
"""
import os
import uuid
from pathlib import Path

import click
import pandas as pd
import sqlalchemy as sa

import mlflow
from mlflow.store.model_registry.dbmodels.models import (
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelTag,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlExperiment,
    SqlExperimentTag,
    SqlLatestMetric,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlTag,
)

TABLES = [
    SqlExperiment.__tablename__,
    SqlRun.__tablename__,
    SqlMetric.__tablename__,
    SqlParam.__tablename__,
    SqlTag.__tablename__,
    SqlExperimentTag.__tablename__,
    SqlLatestMetric.__tablename__,
    SqlRegisteredModel.__tablename__,
    SqlModelVersion.__tablename__,
    SqlRegisteredModelTag.__tablename__,
    SqlModelVersionTag.__tablename__,
]
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


class Model(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return [0]


def log_everything():
    exp_id = mlflow.create_experiment(uuid.uuid4().hex, tags={"tag": "experiment"})
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        mlflow.log_params({"param": "value"})
        mlflow.log_metrics({"metric": 0.1})
        mlflow.set_tags({"tag": "run"})
        model_info = mlflow.pyfunc.log_model(python_model=Model(), artifact_path="model")

    client = mlflow.MlflowClient()
    registered_model_name = uuid.uuid4().hex
    client.create_registered_model(
        registered_model_name, tags={"tag": "registered_model"}, description="description"
    )
    client.create_model_version(
        registered_model_name,
        model_info.model_uri,
        run_id=run.info.run_id,
        tags={"tag": "model_version"},
        run_link="run_link",
        description="description",
    )


def connect_to_mlflow_db():
    return sa.create_engine(os.environ["MLFLOW_TRACKING_URI"]).connect()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--verbose", is_flag=True, default=False)
def pre_migration(verbose):
    for _ in range(5):
        log_everything()
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    with connect_to_mlflow_db() as conn:
        for table in TABLES:
            df = pd.read_sql(sa.text(f"SELECT * FROM {table}"), conn)
            df.to_pickle(SNAPSHOTS_DIR / f"{table}.pkl")
            if verbose:
                click.secho(f"\n{table}\n", fg="blue")
                click.secho(df.head(5).to_markdown(index=False))


@cli.command()
def post_migration():
    with connect_to_mlflow_db() as conn:
        for table in TABLES:
            df_actual = pd.read_sql(sa.text(f"SELECT * FROM {table}"), conn)
            df_expected = pd.read_pickle(SNAPSHOTS_DIR / f"{table}.pkl")
            pd.testing.assert_frame_equal(df_actual[df_expected.columns], df_expected)


if __name__ == "__main__":
    cli()
