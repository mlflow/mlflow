from pathlib import Path

from sqlalchemy import create_engine
from alembic.command import upgrade
from alembic.config import Config


def migrate(url: str, revision: str) -> None:
    alembic_dir = Path(__file__).parent / "db" / "migrations"
    alembic_ini_path = alembic_dir / "alembic.ini"
    url = url.replace("%", "%%")
    config = Config(alembic_ini_path)
    config.set_main_option("script_location", str(alembic_dir))
    config.set_main_option("sqlalchemy.url", url)

    with create_engine(url).begin() as connection:
        config.attributes["connection"] = connection
        upgrade(config, revision)
