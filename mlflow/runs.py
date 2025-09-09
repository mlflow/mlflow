"""
CLI for runs
"""

import json

import click

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus, ViewType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.exceptions import MlflowException
from mlflow.tracking import _get_store
from mlflow.utils.string_utils import _create_table
from mlflow.utils.time import conv_longdate_to_str

RUN_ID = click.option("--run-id", type=click.STRING, required=True)


@click.group("runs")
def commands():
    """
    Manage runs. To manage runs of experiments associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """


@commands.command("list")
@click.option(
    "--experiment-id",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    help="Specify the experiment ID for list of runs.",
    required=True,
)
@click.option(
    "--view",
    "-v",
    default="active_only",
    help="Select view type for list experiments. Valid view types are "
    "'active_only' (default), 'deleted_only', and 'all'.",
)
def list_run(experiment_id: str, view: str) -> None:
    """
    List all runs of the specified experiment in the configured tracking server.
    """
    store = _get_store()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    runs = store.search_runs([experiment_id], None, view_type)
    table = []
    for run in runs:
        run_name = run.info.run_name or ""
        table.append([conv_longdate_to_str(run.info.start_time), run_name, run.info.run_id])
    click.echo(_create_table(sorted(table, reverse=True), headers=["Date", "Name", "ID"]))


@commands.command("delete")
@RUN_ID
def delete_run(run_id: str) -> None:
    """
    Mark a run for deletion. Return an error if the run does not exist or
    is already marked. You can restore a marked run with ``restore_run``,
    or permanently delete a run in the backend store.
    """
    store = _get_store()
    store.delete_run(run_id)
    click.echo(f"Run with ID {run_id} has been deleted.")


@commands.command("restore")
@RUN_ID
def restore_run(run_id: str) -> None:
    """
    Restore a deleted run.
    Returns an error if the run is active or has been permanently deleted.
    """
    store = _get_store()
    store.restore_run(run_id)
    click.echo(f"Run with id {run_id} has been restored.")


@commands.command("describe")
@RUN_ID
def describe_run(run_id: str) -> None:
    """
    All of run details will print to the stdout as JSON format.
    """
    store = _get_store()
    run = store.get_run(run_id)
    json_run = json.dumps(run.to_dictionary(), indent=4)
    click.echo(json_run)


@commands.command("create")
@click.option(
    "--experiment-id",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    help="ID of the experiment under which to create the run. "
    "Must specify either this or --experiment-name.",
)
@click.option(
    "--experiment-name",
    envvar=MLFLOW_EXPERIMENT_NAME.name,
    type=click.STRING,
    help="Name of the experiment under which to create the run. "
    "Must specify either this or --experiment-id.",
)
@click.option(
    "--run-name",
    type=click.STRING,
    help="Optional human-readable name for the run (e.g., 'baseline-model-v1').",
)
@click.option(
    "--description",
    type=click.STRING,
    help="Optional longer description of what this run represents.",
)
@click.option(
    "--tags",
    "-t",
    multiple=True,
    help="Key-value pairs to categorize and filter runs. Use multiple times for "
    "multiple tags. Format: key=value (e.g., env=prod, model=xgboost, version=1.0).",
)
@click.option(
    "--status",
    type=click.Choice(["FINISHED", "FAILED", "KILLED"], case_sensitive=False),
    default="FINISHED",
    help="Final status of the run. Options: FINISHED (default), FAILED, or KILLED.",
)
@click.option(
    "--parent-run-id",
    type=click.STRING,
    help="Optional ID of a parent run to create a nested run under.",
)
def create_run(
    experiment_id: str | None,
    experiment_name: str | None,
    run_name: str | None,
    description: str | None,
    tags: tuple[str, ...],
    status: str,
    parent_run_id: str | None,
) -> None:
    """
    Create a new MLflow run and immediately end it with the specified status.

    This command is useful for creating runs programmatically for testing, scripting,
    or recording completed experiments. The run will be created and immediately closed
    with the specified status (FINISHED, FAILED, or KILLED).
    """
    # Validate that exactly one of experiment_id or experiment_name is provided
    if (experiment_id is not None and experiment_name is not None) or (
        experiment_id is None and experiment_name is None
    ):
        raise click.UsageError("Must specify exactly one of --experiment-id or --experiment-name.")

    # Parse tags from key=value format
    tags_dict = {}
    if tags:
        for tag in tags:
            match tag.split("=", 1):
                case [key, value]:
                    if key in tags_dict:
                        raise click.UsageError(f"Duplicate tag key: '{key}'")
                    tags_dict[key] = value
                case _:
                    raise click.UsageError(
                        f"Invalid tag format: '{tag}'. Tags must be in key=value format."
                    )

    # Set the experiment if using experiment_name
    if experiment_name:
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        experiment_id = experiment.experiment_id

    # Start the run with the specified parameters
    try:
        # Start the run
        active_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=bool(parent_run_id),
            parent_run_id=parent_run_id,
            tags=tags_dict,
            description=description,
        )
        run_id = active_run.info.run_id
        actual_experiment_id = active_run.info.experiment_id

        # End the run with the specified status
        mlflow.end_run(status=RunStatus.to_string(getattr(RunStatus, status.upper())))

        # Output the created run information
        output = {
            "run_id": run_id,
            "experiment_id": actual_experiment_id,
            "status": status.upper(),
            "run_name": run_name,
        }

        click.echo(json.dumps(output, indent=2))

    except MlflowException as e:
        raise click.ClickException(f"Failed to create run: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error creating run: {e!s}")


@commands.command("link-traces")
@click.option(
    "--run-id",
    type=click.STRING,
    required=True,
    help="ID of the run to link traces to.",
)
@click.option(
    "trace_ids",
    "--trace-id",
    "-t",
    multiple=True,
    required=True,
    help="Trace ID to link to the run. Can be specified multiple times (maximum 100 traces).",
)
def link_traces(run_id: str, trace_ids: tuple[str, ...]) -> None:
    """
    Link traces to a run.

    This command links one or more traces to an existing run. Traces can be
    linked to runs to establish relationships between traces and runs.
    Maximum 100 traces can be linked in a single command.
    """
    try:
        client = MlflowClient()
        client.link_traces_to_run(list(trace_ids), run_id)

        # Output success message with count
        click.echo(f"Successfully linked {len(trace_ids)} trace(s) to run '{run_id}'")

    except MlflowException as e:
        raise click.ClickException(f"Failed to link traces: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error linking traces: {e!s}")
