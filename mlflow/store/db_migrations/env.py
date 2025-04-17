from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
from mlflow.store.db.base_sql_model import Base

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    # Try https://stackoverflow.com/questions/30378233/sqlite-lack-of-alter-support-alembic-migration-failing-because-of-this-solutio
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # If available, use a shared connection for the database upgrade, ensuring that any
    # connection-dependent state (e.g., the state of an in-memory database) is preserved
    # for reference by the upgrade routine. For more information, see
    # https://alembic.sqlalchemy.org/en/latest/cookbook.html#sharing-a-
    # connection-with-a-series-of-migration-commands-and-environments
    connection = config.attributes.get("connection")
    if connection is None:
        engine = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    else:
        engine = connection.engine

    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
