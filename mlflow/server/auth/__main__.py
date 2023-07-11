import click

from . import db


@click.group()
def commands():
    pass


@commands.command()
@click.option("--url", required=True)
@click.option("--revision", default="head")
def migrate(url: str, revision: str) -> None:
    db.migrate(url, revision)


if __name__ == "__main__":
    commands()
