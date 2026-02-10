from pathlib import Path

import click

from mlflow.store.fs2db import migrate
from mlflow.store.fs2db._verify import verify_migration


@click.command("migrate-filestore")
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Root directory containing mlruns/ FileStore data.",
)
@click.option(
    "--target",
    required=True,
    help="SQLite URI (e.g. sqlite:///mlflow.db).",
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="After migration, verify that DB row counts match source file counts.",
)
def migrate_filestore(source: str, target: str, verify: bool) -> None:
    """Migrate MLflow FileStore data to a SQLite database."""
    if not target.startswith("sqlite:///"):
        raise click.BadParameter(
            "Must be a SQLite URI starting with 'sqlite:///'",
            param_hint="'--target'",
        )

    migrate(Path(source), target)

    if verify:
        verify_migration(Path(source), target)
