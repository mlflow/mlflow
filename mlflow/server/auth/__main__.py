import click
from pathlib import Path

from sqlalchemy import create_engine
from alembic.command import upgrade
from alembic.config import Config


@click.group()
def commands():
    pass


@commands.command()
@click.option("--url", required=True)
@click.option("--revision", default="head")
def migrate(url: str, revision: str):
    alembic_dir = Path(__file__).parent / "db" / "migrations"
    alembic_ini_path = alembic_dir / "alembic.ini"
    url = url.replace("%", "%%")
    config = Config(alembic_ini_path)
    config.set_main_option("script_location", str(alembic_dir))
    config.set_main_option("sqlalchemy.url", url)

    with create_engine(url).begin() as connection:
        config.attributes["connection"] = connection
        upgrade(config, revision)


if __name__ == "__main__":
    commands()
