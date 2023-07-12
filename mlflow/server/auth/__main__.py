import click
import sqlalchemy

from .db import utils


@click.group()
def commands():
    pass


@commands.command()
@click.option("--url", required=True)
@click.option("--revision", default="head")
def migrate(url: str, revision: str) -> None:
    with sqlalchemy.create_engine(url).begin() as engine:
        utils.migrate(engine, revision)


if __name__ == "__main__":
    commands()
