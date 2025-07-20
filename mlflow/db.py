import click


@click.group("db")
def commands():
    """
    Commands for managing an MLflow tracking database.
    """


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
    import mlflow.store.db.utils

    engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(url)
    
    # Check if the database has been initialized with the base schema
    if mlflow.store.db.utils._all_tables_exist(engine):
        # Database already has tables, just run any pending migrations
        mlflow.store.db.utils._upgrade_db(engine)
    else:
        # Database is fresh or missing core tables, initialize it properly
        # This will create the initial schema and run all migrations
        mlflow.store.db.utils._initialize_tables(engine)
