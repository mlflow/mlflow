import click


@click.group("db")
def commands():
    """
    Commands for managing an mlflux tracking database.
    """
    pass


@commands.command()
@click.argument("url")
def upgrade(url):
    """
    Upgrade the schema of an mlflux tracking database to the latest supported version.

    **IMPORTANT**: Schema migrations can be slow and are not guaranteed to be transactional -
    **always take a backup of your database before running migrations**. The migrations README,
    which is located at
    https://github.com/mlflux/mlflux/blob/master/mlflux/store/db_migrations/README.md, describes
    large migrations and includes information about how to estimate their performance and
    recover from failures.
    """
    import mlflux.store.db.utils

    engine = mlflux.store.db.utils.create_sqlalchemy_engine_with_retry(url)
    mlflux.store.db.utils._upgrade_db(engine)
