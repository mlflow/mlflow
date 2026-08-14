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
    if mlflow.store.db.utils._is_empty_database(engine):
        mlflow.store.db.utils._initialize_tables(engine)
    else:
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
    import sqlalchemy.exc

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
    except sqlalchemy.exc.SQLAlchemyError as e:
        raise click.ClickException(f"Database error: {e}") from e
    finally:
        if engine is not None:
            engine.dispose()


def _parse_tag(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise click.BadParameter(
            f"Tag {value!r} must be in key=value format (e.g. --tag team=team-a)."
        )
    key, _, val = value.partition("=")
    if not key:
        raise click.BadParameter(f"Tag {value!r} has an empty key. Use key=value format.")
    return key, val


@commands.command("move-resources")
@click.argument("url")
@click.option(
    "--from",
    "source_workspace",
    required=True,
    help="Source workspace name.",
)
@click.option(
    "--to",
    "target_workspace",
    required=True,
    help="Target workspace name.",
)
@click.option(
    "--resource-type",
    required=True,
    help="Table name of the resource type to move (e.g. experiments, registered_models).",
)
@click.option(
    "--name",
    multiple=True,
    help="Resource name(s) to move. Repeatable.",
)
@click.option(
    "--tag",
    multiple=True,
    help=(
        "Tag filter as key=value. Repeatable. "
        "When multiple tags are given, only resources matching ALL tags are included."
    ),
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    show_default=True,
    help="Show what would be moved without making changes.",
)
@click.option(
    "--artifact-policy",
    type=click.Choice(["preserve", "retarget"]),
    default="preserve",
    show_default=True,
    help=(
        "How to handle experiment artifact locations (experiments only). 'preserve' keeps "
        "them unchanged. 'retarget' repoints every moved experiment to the target "
        "workspace's artifact root. Artifact objects and stored run, logged model and "
        "trace URIs are not modified."
    ),
)
@click.option(
    "--default-artifact-root",
    default=None,
    help=(
        "Artifact root the tracking server is started with. Used by --artifact-policy "
        "retarget when the target workspace does not define its own default_artifact_root."
    ),
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
def move_resources(
    url,
    source_workspace,
    target_workspace,
    resource_type,
    name,
    tag,
    dry_run,
    artifact_policy,
    default_artifact_root,
    verbose,
    yes,
):
    """
    Move resources from one workspace to another.

    Selectively move workspace-scoped resources between workspaces by name
    or tag filter (mutually exclusive). When neither --name nor --tag is
    specified, all resources of the given type in the source workspace are moved.

    The --resource-type value is the database table name (e.g. experiments,
    registered_models, evaluation_datasets, webhooks, jobs).

    Tag filtering (--tag) is supported for experiments and registered_models
    only. When multiple --tag flags are given, only resources matching ALL tags
    are included (AND logic).

    \b
    Examples:
      # Move specific experiments by name
      mlflow db move-resources sqlite:///mlflow.db \\
        --from default --to team-a --resource-type experiments \\
        --name training-v1 --name training-v2
      # Move experiments matching ALL specified tags
      mlflow db move-resources sqlite:///mlflow.db \\
        --from default --to team-a --resource-type experiments \\
        --tag team=team-a --tag env=prod
      # Move all registered models from one workspace to another
      mlflow db move-resources sqlite:///mlflow.db \\
        --from default --to team-a --resource-type registered_models
      # Move experiments and repoint their artifact roots to the target
      # workspace's artifact root (new runs land there, existing artifact
      # URIs are not modified)
      mlflow db move-resources sqlite:///mlflow.db \\
        --from default --to team-a --resource-type experiments \\
        --name training-v1 --artifact-policy retarget \\
        --default-artifact-root s3://mlflow-artifacts

    With --artifact-policy retarget (experiments only), every moved experiment's
    artifact_location is repointed to the artifact root resolved for the target
    workspace, in the same transaction as the move. Artifact objects are not
    copied or deleted, and stored run, logged model and trace URIs are left
    unchanged, so everything already logged keeps resolving at its current
    location while new runs land under the new root.

    **IMPORTANT**: Always take a backup of your database before running this command.
    """
    import sqlalchemy as sa
    import sqlalchemy.exc

    import mlflow.store.db.utils
    from mlflow.exceptions import MlflowException
    from mlflow.store.db.workspace_move import RESOURCE_TYPE_CHOICES
    from mlflow.store.db.workspace_move import move_resources as move
    from mlflow.store.db.workspace_utils import _NOT_ENABLED_MSG, format_truncated_list
    from mlflow.tracking._workspace.registry import get_workspace_store
    from mlflow.utils.workspace_utils import resolve_workspace_store_uri

    if resource_type not in RESOURCE_TYPE_CHOICES:
        raise click.ClickException(
            f"Unknown resource type {resource_type!r}. "
            f"Valid types: {', '.join(RESOURCE_TYPE_CHOICES)}"
        )

    parsed_tags = [_parse_tag(t) for t in tag] if tag else None
    parsed_names = list(name) if name else None

    engine = None
    try:
        engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(url)
        needs_confirmation = not dry_run and not yes

        # Probe before constructing the workspace store: on database URIs the
        # provider initializes tables on an empty database, and a mistargeted
        # URL should fail cleanly instead.
        if not sa.inspect(engine).has_table("workspaces"):
            raise RuntimeError(_NOT_ENABLED_MSG)

        workspace_store = get_workspace_store(resolve_workspace_store_uri(None, tracking_uri=url))

        result = move(
            engine,
            workspace_store,
            source_workspace=source_workspace,
            target_workspace=target_workspace,
            resource_type=resource_type,
            names=parsed_names,
            tags=parsed_tags,
            dry_run=dry_run or needs_confirmation,
            verbose=verbose,
            artifact_policy=artifact_policy,
            default_artifact_root=default_artifact_root,
        )

        if not result.names:
            click.echo(f"No {resource_type} to move.")
            return

        max_display = None if verbose else 20
        name_list = format_truncated_list(result.names, max_rows=max_display)

        extra_notes: list[str] = []
        if result.row_count > len(result.names):
            extra_notes.append(
                f"Note: {result.row_count} rows match {len(result.names)} distinct "
                f"name(s). All rows with a matching name will be moved."
            )

        if dry_run:
            click.echo(
                f"Dry run completed. {result.row_count} {resource_type} row(s) would be moved "
                f"from {source_workspace!r} to {target_workspace!r}:{name_list}"
            )
            for note in extra_notes:
                click.echo(note)
            if result.retarget_root:
                click.echo(f"Artifact roots would be repointed under {result.retarget_root}.")
            return

        if needs_confirmation:
            click.echo(
                f"{result.row_count} {resource_type} row(s) to move from "
                f"{source_workspace!r} to {target_workspace!r}:{name_list}"
            )
            for note in extra_notes:
                click.echo(note)
            if result.retarget_root:
                click.echo(f"Artifact roots would be repointed under {result.retarget_root}.")
            click.confirm("Proceed with move?", default=False, abort=True)
            # Re-run the full move (including conflict detection) in a new
            # transaction. The preview counts above may differ from the
            # actual move if another admin modified the data in between,
            # but the second call is self-consistent and safe.
            result = move(
                engine,
                workspace_store,
                source_workspace=source_workspace,
                target_workspace=target_workspace,
                resource_type=resource_type,
                names=parsed_names,
                tags=parsed_tags,
                dry_run=False,
                verbose=verbose,
                artifact_policy=artifact_policy,
                default_artifact_root=default_artifact_root,
            )

        click.echo(
            f"Moved {result.row_count} {resource_type} row(s) "
            f"from {source_workspace!r} to {target_workspace!r}."
        )
        if result.retarget_root:
            click.echo(
                f"Repointed {result.row_count} experiment artifact root(s) "
                f"under {result.retarget_root}."
            )
    except (RuntimeError, MlflowException) as e:
        raise click.ClickException(str(e)) from e
    except sqlalchemy.exc.SQLAlchemyError as e:
        raise click.ClickException(f"Database error: {e}") from e
    finally:
        if engine is not None:
            engine.dispose()
