"""
CLI for runs
"""
import sys
import click
import json
import mlflow.tracking
from mlflow.entities import ViewType
from mlflow.tracking import _get_store
from tabulate import tabulate
from mlflow.entities.experiment import Experiment
from mlflow.utils.time_utils import conv_longdate_to_str

RUN_ID = click.option("--run-id", type=click.STRING, required=True)


@click.group("runs")
def commands():
    """
    Manage runs. To manage runs of experiments associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """
    pass


@commands.command("list")
@click.option("--experiment-id", envvar=mlflow.tracking._EXPERIMENT_ID_ENV_VAR, type=click.INT,
              help="Specify the experiment ID for list of runs Defaults to %s" %
                   Experiment.DEFAULT_EXPERIMENT_ID, default=Experiment.DEFAULT_EXPERIMENT_ID)
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
    table = [[conv_longdate_to_str(run.start_time), run.name, run.run_uuid] for run in runs]
    print(tabulate(sorted(table), headers=["Date", "Name", "ID"]))


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
def restore_run(run_id):
    """
    Restore a deleted run.
    Returns an error if the run is active or has been permanently deleted.
    """
    store = _get_store()
    store.restore_run(run_id)
    print("Run with id %s has been restored." % str(run_id))


@commands.command('describe')
@RUN_ID
def describe_run(run_id):
    """
    All of run details will print to the stdout as JSON format.
    """
    store = _get_store()
    run = store.get_run(run_id)
    json_run = json.dumps(run.to_dictionary(), indent=4)
    print(json_run, file=sys.stdout)
