"""
CLI for runs
"""

import click
import datetime
import json
from mlflow.entities import ViewType
from mlflow.tracking import _get_store
from tabulate import tabulate

RUN_ID = click.argument("run_id", type=click.STRING)

@click.group("runs")
def commands():
    """
    Manage runs. To manage runs associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """
    pass

@commands.command("list")
@click.option("--experiment_id", help="Specify the experiment for list of runs.")
@click.option("--view", "-v", default="active_only",
              help="Select view type for list experiments. Valid view types are "
                   "'active_only' (default), 'deleted_only', and 'all'.")
def list_run(experiment_id, view):
    """
    List all runs of the specified experiment in the configured tracking server.
    """
    store = _get_store()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    runs = store.list_run_infos(experiment_id, view_type)
    table = [[run.run_uuid, run.name, datetime.datetime.fromtimestamp(run.start_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')] for run in runs]
    print(tabulate(sorted(table), headers=["Run Id", "Name", "Start time"]))


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

@commands.command('export')
@RUN_ID
@click.argument('file', type=click.Path(exists=False))
def export_run(run_id, file):
    """
    Export a run to JSON file.
    """
    store = _get_store()
    run = store.get_run(run_id)

    with open(file, 'w') as outfile:
        json.dump(run.to_dictionary(), outfile, indent=4)
    print(f'Run with id {run_id} saved in {file}')