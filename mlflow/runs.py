"""
CLI for runs
"""

import click
import datetime
from mlflow.entities import ViewType
from mlflow.tracking import _get_store
from tabulate import tabulate

EXPERIMENT_ID = click.argument("experiment_id", type=click.INT)
RUN_ID = click.argument("run_id", type=click.INT)

@click.group("runs")
def commands():
    """
    Manage runs. To manage runs associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """
    pass

@commands.command("list")
@click.option("--view", "-v", default="active_only",
              help="Select view type for list experiments. Valid view types are "
                   "'active_only' (default), 'deleted_only', and 'all'.")
@EXPERIMENT_ID
def list_experiments(experiment_id, view):
    """
    List all runs of the specified experiment in the configured tracking server.
    """
    store = _get_store()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    runs = store.list_run_infos(experiment_id, view_type)
    table = [[run.run_uuid, run.name, datetime.datetime.fromtimestamp(run.start_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')] for run in runs]
    print(tabulate(sorted(table), headers=["Run Id", "Name", "Start time"]))


@commands.command("delete")
@EXPERIMENT_ID
def delete_experiment(run_id):
    """
    Mark a run for deletion. Return an error if the run does not exist or
    is already marked. You can restore a marked run with ``restore_run``,
    or permanently delete a run in the backend store.
    """
    store = _get_store()
    store.delete_run(run_id)
    print("Run with ID %s has been deleted." % str(run_id))

@commands.command("restore")
@RUN_ID
def restore_experiment(run_id):
    """
    Restore a deleted run.
    Returns an error if the run is active or has been permanently deleted.
    """
    store = _get_store()
    store.restore_run(run_id)
    print("Run with id %s has been restored." % str(run_id))
