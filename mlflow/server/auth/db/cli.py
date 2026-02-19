import click
import sqlalchemy

from mlflow.server.auth.db import utils


@click.group(name="db")
def commands():
    pass


@commands.command()
@click.option("--url", required=True)
@click.option("--revision", default="head")
def upgrade(url: str, revision: str) -> None:
    engine = sqlalchemy.create_engine(url)
    utils.migrate(engine, revision)
    engine.dispose()
