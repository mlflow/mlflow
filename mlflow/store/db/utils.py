import os

from contextlib import contextmanager
import logging

from alembic.migration import MigrationContext  # pylint: disable=import-error
from alembic.script import ScriptDirectory
import sqlalchemy

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.store.db.db_types import SQLITE

_logger = logging.getLogger(__name__)


MLFLOW_SQLALCHEMYSTORE_POOL_SIZE = "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE"
MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW = "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW"


def _get_package_dir():
    """Returns directory containing MLflow python package."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir))


def _initialize_tables(engine):
    _logger.info("Creating initial MLflow database tables...")
    InitialBase.metadata.create_all(engine)
    _upgrade_db(engine)


def _get_latest_schema_revision():
    """Get latest schema revision as a string."""
    # We aren't executing any commands against a DB, so we leave the DB URL unspecified
    config = _get_alembic_config(db_url="")
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    if len(heads) != 1:
        raise MlflowException("Migration script directory was in unexpected state. Got %s head "
                              "database versions but expected only 1. Found versions: %s"
                              % (len(heads), heads))
    return heads[0]


def _verify_schema(engine):
    head_revision = _get_latest_schema_revision()
    current_rev = _get_schema_version(engine)
    if current_rev != head_revision:
        raise MlflowException(
            "Detected out-of-date database schema (found version %s, but expected %s). "
            "Take a backup of your database, then run 'mlflow db upgrade <database_uri>' "
            "to migrate your database to the latest schema. NOTE: schema migration may "
            "result in database downtime - please consult your database's documentation for "
            "more detail." % (current_rev, head_revision))


def _get_managed_session_maker(SessionMaker, db_type):
    """
    Creates a factory for producing exception-safe SQLAlchemy sessions that are made available
    using a context manager. Any session produced by this factory is automatically committed
    if no exceptions are encountered within its associated context. If an exception is
    encountered, the session is rolled back. Finally, any session produced by this factory is
    automatically closed when the session's associated context is exited.
    """

    @contextmanager
    def make_managed_session():
        """Provide a transactional scope around a series of operations."""
        session = SessionMaker()
        try:
            if db_type == SQLITE:
                session.execute("PRAGMA foreign_keys = ON;")
                session.execute("PRAGMA case_sensitive_like = true;")
            yield session
            session.commit()
        except MlflowException:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise MlflowException(message=e, error_code=INTERNAL_ERROR)
        finally:
            session.close()

    return make_managed_session


def _get_alembic_config(db_url, alembic_dir=None):
    """
    Constructs an alembic Config object referencing the specified database and migration script
    directory.

    :param db_url Database URL, like sqlite:///<absolute-path-to-local-db-file>. See
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls for a full list of valid
    database URLs.
    :param alembic_dir Path to migration script directory. Uses canonical migration script
    directory under mlflow/alembic if unspecified. TODO: remove this argument in MLflow 1.1, as
    it's only used to run special migrations for pre-1.0 users to remove duplicate constraint
    names.
    """
    from alembic.config import Config
    final_alembic_dir = os.path.join(_get_package_dir(), 'store', 'db_migrations')\
        if alembic_dir is None else alembic_dir
    # Escape any '%' that appears in a db_url. This could be in a password,
    # url, or anything that is part of a potentially complex database url
    db_url = db_url.replace('%', '%%')
    config = Config(os.path.join(final_alembic_dir, 'alembic.ini'))
    config.set_main_option('script_location', final_alembic_dir)
    config.set_main_option('sqlalchemy.url', db_url)
    return config


def _upgrade_db(engine):
    """
    Upgrade the schema of an MLflow tracking database to the latest supported version.
    Note that schema migrations can be slow and are not guaranteed to be transactional -
    we recommend taking a backup of your database before running migrations.

    :param url Database URL, like sqlite:///<absolute-path-to-local-db-file>. See
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls for a full list of valid
    database URLs.
    """
    # alembic adds significant import time, so we import it lazily
    from alembic import command
    db_url = str(engine.url)
    _logger.info("Updating database tables")
    config = _get_alembic_config(db_url)
    # Initialize a shared connection to be used for the database upgrade, ensuring that
    # any connection-dependent state (e.g., the state of an in-memory database) is preserved
    # for reference by the upgrade routine. For more information, see
    # https://alembic.sqlalchemy.org/en/latest/cookbook.html#sharing-a-
    # connection-with-a-series-of-migration-commands-and-environments
    with engine.begin() as connection:
        config.attributes['connection'] = connection  # pylint: disable=E1137
        command.upgrade(config, 'heads')


def _get_schema_version(engine):
    with engine.connect() as connection:
        mc = MigrationContext.configure(connection)
        return mc.get_current_revision()


def _is_initialized_before_mlflow_1(engine):
    """
    Returns true if the database at the specified URL was initialized before MLflow 1.0, False
    otherwise.
    A database is initialized before MLflow 1.0 if and only if its revision ID is set to None.
    """
    return _get_schema_version(engine) is None


def _upgrade_db_initialized_before_mlflow_1(engine):
    """
    Upgrades the schema of an MLflow tracking database created prior to MLflow 1.0, removing
    duplicate constraint names. This method performs a one-time update for pre-1.0 users that we
    plan to make available in MLflow 1.0 but remove in successive versions (e.g. MLflow 1.1),
    after which we will assume that effectively all databases have been initialized using the schema
    in mlflow.store.dbmodels.initial_models (with a small number of special-case databases
    initialized pre-1.0 and migrated to have the same schema as mlflow.store.dbmodels.initial_models
    via this method).
    TODO: remove this method in MLflow 1.1.
    """
    # alembic adds significant import time, so we import it lazily
    from alembic import command
    _logger.info("Updating database tables in preparation for MLflow 1.0 schema migrations ")
    alembic_dir = os.path.join(_get_package_dir(), 'temporary_db_migrations_for_pre_1_users')
    config = _get_alembic_config(str(engine.url), alembic_dir)
    command.upgrade(config, 'heads')
    # Reset the alembic version to "base" (the 'first' version) so that a) the versioning system
    # is unaware that this migration occurred and b) subsequent migrations, like the migration to
    # add metric steps, do not need to depend on this one. This allows us to eventually remove this
    # method and the associated migration e.g. in MLflow 1.1.
    command.stamp(config, "base")


def create_sqlalchemy_engine(db_uri):
    pool_size = os.environ.get(MLFLOW_SQLALCHEMYSTORE_POOL_SIZE)
    pool_max_overflow = os.environ.get(MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW)
    pool_kwargs = {}
    # Send argument only if they have been injected.
    # Some engine does not support them (for example sqllite)
    if pool_size:
        pool_kwargs['pool_size'] = int(pool_size)
    if pool_max_overflow:
        pool_kwargs['max_overflow'] = int(pool_max_overflow)
    if pool_kwargs:
        _logger.info("Create SQLAlchemy engine with pool options %s", pool_kwargs)
    return sqlalchemy.create_engine(db_uri, pool_pre_ping=True,
                                    **pool_kwargs)
