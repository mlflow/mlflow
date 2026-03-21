import click


@click.group("db")
def commands():
    """
    Commands for managing an MLflow tracking database.
    """


@commands.command()
@click.argument("url")
def upgrade(url):
    """
    Upgrade the schema of an MLflow tracking database to the latest supported version.

    **IMPORTANT**: Schema migrations can be slow and are not guaranteed to be transactional -
    **always take a backup of your database before running migrations**. The migrations README,
    which is located at
    https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/README.md, describes
    large migrations and includes information about how to estimate their performance and
    recover from failures.
    """
    import mlflow.store.db.utils

    engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(url)
    mlflow.store.db.utils._upgrade_db(engine)


@commands.command("migrate-to-default-workspace")
@click.argument("url")
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    show_default=True,
    help="Check for conflicts and report how many rows would be moved.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="List all conflicts instead of truncating the output.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
def migrate_to_default_workspace(url, dry_run, verbose, yes):
    """
    Move workspace-scoped resources into the default workspace.

    **IMPORTANT**: This operation runs in a single transaction, but can still be long-running.
    Always take a backup of your database before running this command.
    """
    import mlflow.store.db.utils
    from mlflow.store.db.workspace_migration import migrate_to_default_workspace as migrate

    engine = None
    try:
        engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(url)
        counts = migrate(engine, dry_run=True, verbose=verbose)

        total = sum(counts.values())
        if dry_run:
            click.echo("Dry run completed. Rows that would be moved to the default workspace:")
            for table_name, count in counts.items():
                click.echo(f"  {table_name}: {count}")
            click.echo(f"Total rows: {total}")
            return

        if total == 0:
            click.echo("No rows need to be moved.")
            return

        click.echo("Rows to be moved to the default workspace:")
        for table_name, count in counts.items():
            click.echo(f"  {table_name}: {count}")
        click.echo(f"Total rows: {total}")

        if not yes:
            click.confirm("Proceed with migration?", default=False, abort=True)

        migrate(engine, dry_run=False, verbose=verbose)
        click.echo(f"Moved {total} rows to the default workspace.")
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    finally:
        if engine is not None:
            engine.dispose()
