from pathlib import Path

import click

from mlflow.store.fs2db import migrate
from mlflow.utils.uri import get_uri_scheme


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
    "--progress/--no-progress",
    default=True,
    help="Show per-experiment progress messages during migration.",
)
def migrate_filestore(source: str, target: str, progress: bool) -> None:
    """Migrate MLflow FileStore data to a SQLite database."""
    if get_uri_scheme(target) != "sqlite":
        raise click.BadParameter(
            "Must be a SQLite URI starting with 'sqlite:///'",
            param_hint="'--target'",
        )

    db_path = Path(target.removeprefix("sqlite:///")).resolve()
    if not db_path.parent.is_dir():
        raise click.BadParameter(
            f"Parent directory does not exist: {db_path.parent}",
            param_hint="'--target'",
        )
    if db_path.exists():
        click.confirm(f"Database file already exists: {db_path}\nOverwrite?", abort=True)
        db_path.unlink()

    target = f"sqlite:///{db_path}"
    migrate(Path(source), target, progress=progress)
