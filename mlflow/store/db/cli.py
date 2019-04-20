import click
import logging


_logger = logging.getLogger(__name__)

@click.group("db")
def commands():
    """
    Upload, list, and download artifacts from an MLflow artifact repository.

    To manage artifacts for a run associated with a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@click.argument("url")
def upgrade(url):
    # alembic adds significant import time, so we import it lazily
    from alembic import command
    from alembic.config import Config

    _logger.info("Updating database tables")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.normpath(os.path.join(current_dir, '..'))
    directory = os.path.join(package_dir, 'migrations')
    config = Config(os.path.join(package_dir, 'alembic.ini'))
    # Taken from https://github.com/apache/airflow/blob/6970b233964ee254bbb343ed8bdc906c2f7bd974/airflow/utils/db.py#L301
    config.set_main_option('script_location', directory.replace('%', '%%'))
    config.set_main_option('sqlalchemy.url', url)
    command.upgrade(config, 'heads')
