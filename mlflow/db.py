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
    version. Note that schema migrations can be slow and are not guaranteed to be transactional -
    always take a backup of your database before running migrations.
    """
    if mlflow.store.db.utils._is_initialized_before_mlflow_1(url):
        mlflow.store.db.utils._upgrade_db_initialized_before_mlflow_1(url)
    mlflow.store.db.utils._upgrade_db(url)
