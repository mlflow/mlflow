import os

import logging


_logger = logging.getLogger(__name__)


def _get_alembic_config(db_url):
    from alembic.config import Config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir))
    directory = os.path.join(package_dir, 'alembic')
    config = Config(os.path.join(package_dir, 'alembic.ini'))
    config.set_main_option('script_location', directory)
    config.set_main_option('sqlalchemy.url', db_url)
    return config


def _upgrade_db(url):
    """
    Upgrade the schema of an MLflow tracking database to the latest supported version.
    version. Note that schema migrations can be slow and are not guaranteed to be transactional -
    we recommend taking a backup of your database before running migrations.

    :param url Database URL, like sqlite:///<absolute-path-to-local-db-file>. See
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls for a full list of valid
    database URLs.
    """
    # alembic adds significant import time, so we import it lazily
    from alembic import command

    _logger.info("Updating database tables at %s", url)
    config = _get_alembic_config(url)
    command.upgrade(config, 'heads')
