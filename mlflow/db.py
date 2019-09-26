import click
import time

import mlflow.store.db.utils
from mlflow.store.sqlalchemy_store import SqlAlchemyStore


@click.group("db")
def commands():
    """
    Commands for managing an MLflow tracking database.
    """
    pass


@commands.command()
@click.argument("url")
def upgrade(url):
    """
    Upgrade the schema of an MLflow tracking database to the latest supported version.
    version.

    **IMPORTANT**: Schema migrations can be slow and are not guaranteed to be transactional -
    **always take a backup of your database before running migrations**. The migrations README,
    which is located at
    https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/README, describes
    large migrations and includes information about how to estimate their performance and
    recover from failures.
    """
    if mlflow.store.db.utils._is_initialized_before_mlflow_1(url):
        mlflow.store.db.utils._upgrade_db_initialized_before_mlflow_1(url)
    mlflow.store.db.utils._upgrade_db(url)


@commands.command()
@click.argument("url")
@click.option("--retention-time", default=None, type=click.INT,
              help="Retention time of metrics in timestamp millisecond. Metric from runs older"
                   "than retention_time will be sampled.")
@click.option("--nb-metrics-to-keep", default=None, type=click.INT,
              help="Number of point to keep for each different sampled metrics.")
def cleanup(url, retention_time, nb_metrics_to_keep):
    """
    Cleanup the database by sampling the metrics from runs older that retention_time.
    The metrics are sampled by keeping nb_metrics_to_keep point for each individual metric.
    """
    db_store = SqlAlchemyStore(db_uri=url, default_artifact_root="/tmp")
    max_timestamp = int(time.time()*1000) - retention_time
    db_store.sample_metrics_in_interval(None, max_timestamp, nb_metrics_to_keep)
