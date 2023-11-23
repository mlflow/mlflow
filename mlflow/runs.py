"""
CLI for runs
"""
import json

import click

from mlflow.entities import ViewType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
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
    pass


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
def list_run(experiment_id, view):
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
def delete_run(run_id):
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
def restore_run(run_id):
    """
    Restore a deleted run.
    Returns an error if the run is active or has been permanently deleted.
    """
    store = _get_store()
    store.restore_run(run_id)
    click.echo(f"Run with id {run_id} has been restored.")


@commands.command("describe")
@RUN_ID
def describe_run(run_id):
    """
    All of run details will print to the stdout as JSON format.
    """
    store = _get_store()
    run = store.get_run(run_id)
    json_run = json.dumps(run.to_dictionary(), indent=4)
    click.echo(json_run)
