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


@commands.command("check-upgrade")
@click.argument("url")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output results as JSON.",
)
def check_upgrade(url, output_json):
    """
    Check pending schema migrations and classify them by safety level.

    Analyzes the upgrade path from the database's current schema version to the latest
    version and reports whether each migration is SAFE (additive), CAUTIOUS (needs review),
    or BREAKING (requires downtime).

    Exit codes: 0 = all safe, 1 = cautious migrations present, 2 = breaking migrations present.
    """
    import json
    import sys

    import mlflow.store.db.utils
    from mlflow.store.db_migrations.migration_classifier import (
        MigrationSafety,
        classify_range,
    )

    engine = None
    try:
        engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(url)
        current_rev = mlflow.store.db.utils._get_schema_version(engine)
        head_rev = mlflow.store.db.utils._get_latest_schema_revision()

        if current_rev == head_rev:
            if output_json:
                click.echo(json.dumps({"status": "up_to_date", "current": current_rev}))
            else:
                click.echo(f"Database schema is up to date (revision {current_rev}).")
            sys.exit(0)

        analyses = classify_range(current_rev, head_rev)

        if output_json:
            result = {
                "current_revision": current_rev,
                "head_revision": head_rev,
                "pending_migrations": [
                    {
                        "revision": a.revision,
                        "safety": a.safety.value,
                        "operations": [
                            {"name": op.name, "safety": op.safety.value, "detail": op.detail}
                            for op in a.operations
                        ],
                        "notes": a.notes,
                    }
                    for a in analyses
                ],
            }
            worst = MigrationSafety.SAFE
            for a in analyses:
                if a.safety == MigrationSafety.BREAKING:
                    worst = MigrationSafety.BREAKING
                    break
                if a.safety == MigrationSafety.CAUTIOUS:
                    worst = MigrationSafety.CAUTIOUS
            result["overall_safety"] = worst.value
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Current revision: {current_rev}")
            click.echo(f"Target revision:  {head_rev}")
            click.echo(f"Pending migrations: {len(analyses)}")
            click.echo()

            for a in analyses:
                icon = {"safe": "+", "cautious": "~", "breaking": "!"}[a.safety.value]
                click.echo(f"  [{icon}] {a.revision} ({a.safety.value.upper()})")
                for op in a.operations:
                    click.echo(f"      {op.name}: {op.detail}")
                for note in a.notes:
                    click.echo(f"      note: {note}")

            click.echo()

        worst = MigrationSafety.SAFE
        for a in analyses:
            if a.safety == MigrationSafety.BREAKING:
                worst = MigrationSafety.BREAKING
                break
            if a.safety == MigrationSafety.CAUTIOUS:
                worst = MigrationSafety.CAUTIOUS

        if worst == MigrationSafety.SAFE:
            if not output_json:
                click.echo("All pending migrations are SAFE. Zero-downtime upgrade is possible.")
            sys.exit(0)
        elif worst == MigrationSafety.CAUTIOUS:
            if not output_json:
                click.echo(
                    "Some migrations require CAUTION. Review before upgrading. "
                    "Set MLFLOW_ALLOW_SCHEMA_MISMATCH=true to allow startup "
                    "with pending migrations."
                )
            sys.exit(1)
        else:
            if not output_json:
                click.echo(
                    "BREAKING migrations detected. Downtime is required. "
                    "Use the traditional scale-down -> migrate -> scale-up workflow."
                )
            sys.exit(2)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if engine is not None:
            engine.dispose()


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
