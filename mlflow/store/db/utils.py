import os

import logging


_logger = logging.getLogger(__name__)


def upgrade_db(url):
    """
    Updates a database associated with an MLflow tracking server to the latest expected schema.
    """
    # alembic adds significant import time, so we import it lazily
    from alembic import command
    from alembic.config import Config

    _logger.info("Updating database tables at %s" % url)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.normpath(os.path.join(current_dir, '..', '..', '..'))
    directory = os.path.join(package_dir, 'alembic')
    config = Config(os.path.join(package_dir, 'alembic.ini'))
    config.set_main_option('script_location', directory)
    config.set_main_option('sqlalchemy.url', url)
    command.upgrade(config, 'heads')

