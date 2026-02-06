from pathlib import Path

import click


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
def migrate_filestore(source: str, target: str) -> None:
    """Migrate MLflow FileStore data to a SQLite database."""
    if not target.startswith("sqlite:///"):
        raise click.BadParameter(
            "Must be a SQLite URI starting with 'sqlite:///'",
            param_hint="'--target'",
        )

    from mlflow.store.fs2db import migrate

    migrate(Path(source), target)
