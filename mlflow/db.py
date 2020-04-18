import click

import mlflow.store.db.utils


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

    **IMPORTANT**: Schema migrations can be slow and are not guaranteed to be transactional -
    **always take a backup of your database before running migrations**. The migrations README,
    which is located at
    https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/README.md, describes
    large migrations and includes information about how to estimate their performance and
    recover from failures.
    """
    engine = mlflow.store.db.utils.create_sqlalchemy_engine(url)
    if mlflow.store.db.utils._is_initialized_before_mlflow_1(engine):
        mlflow.store.db.utils._upgrade_db_initialized_before_mlflow_1(engine)
    mlflow.store.db.utils._upgrade_db(engine)
