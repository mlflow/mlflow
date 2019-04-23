import os

import click
import logging


_logger = logging.getLogger(__name__)

@click.group("db")
def commands():
    """
    Commands for updating a database associated with an MLflow tracking server.
    """
    pass


@commands.command()
@click.argument("url")
def upgrade(url):
    """
    Upgrade an MLflow tracking database.

    :param url Database URL, like sqlite:///<absolute-path-to-local-db-file>. See
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls for a full list of valid
    database URLs.
    """
    # alembic adds significant import time, so we import it lazily
    from alembic import command
    from alembic.config import Config

    _logger.info("Updating database tables")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.normpath(os.path.join(current_dir, '..', '..', '..'))
    print(package_dir)
    directory = os.path.join(package_dir, 'alembic')
    config = Config(os.path.join(package_dir, 'alembic.ini'))
    # Taken from https://github.com/apache/airflow/blob/6970b233964ee254bbb343ed8bdc906c2f7bd974/airflow/utils/db.py#L301
    config.set_main_option('script_location', directory.replace('%', '%%'))
    config.set_main_option('sqlalchemy.url', url)
    command.upgrade(config, 'heads')
