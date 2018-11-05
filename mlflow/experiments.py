from __future__ import print_function

import os

import click
from tabulate import tabulate

from mlflow.data import is_uri
from mlflow.entities import ViewType
from mlflow.tracking import _get_store

EXPERIMENT_ID = click.argument("experiment_id", type=click.INT)


@click.group("experiments")
def commands():
    """
    Manage experiments. To manage experiments associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """
    pass


@commands.command()
@click.argument("experiment_name")
@click.option("--artifact-location", "-l",
              help="Base location for runs to store artifact results. Artifacts will be stored "
                   "at $artifact_location/$run_id/artifacts. See "
                   "https://mlflow.org/docs/latest/tracking.html#where-runs-get-recorded for "
                   "more info on the properties of artifact location. "
                   "If no location is provided, the tracking server will pick a default.")
def create(experiment_name, artifact_location):
    """
    Create an experiment in the configured tracking server.
    """
    store = _get_store()
    exp_id = store.create_experiment(experiment_name, artifact_location)
    print("Created experiment '%s' with id %d" % (experiment_name, exp_id))


@commands.command("list")
@click.option("--view", "-v", default="active_only",
              help="Select view type for list experiments. Valid view types are "
                   "'active_only' (default), 'deleted_only', and 'all'.")
def list_experiments(view):
    """
    List all experiments in the configured tracking server.
    """
    store = _get_store()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    experiments = store.list_experiments(view_type)
    table = [[exp.experiment_id, exp.name, exp.artifact_location if is_uri(exp.artifact_location)
              else os.path.abspath(exp.artifact_location)] for exp in experiments]
    print(tabulate(sorted(table), headers=["Experiment Id", "Name", "Artifact Location"]))


@commands.command("delete")
@EXPERIMENT_ID
def delete_experiment(experiment_id):
    """
    Mark an experiment for deletion. Return an error if the experiment does not exist or
    is already marked. You can restore a marked experiment with ``restore_experiment``,
    or permanently delete an experiment in the backend store.
    """
    store = _get_store()
    store.delete_experiment(experiment_id)
    print("Experiment with ID %s has been deleted." % str(experiment_id))


@commands.command("restore")
@EXPERIMENT_ID
def restore_experiment(experiment_id):
    """
    Restore a deleted experiment.
    Returns an error if the experiment is active or has been permanently deleted.
    """
    store = _get_store()
    store.restore_experiment(experiment_id)
    print("Experiment with id %s has been restored." % str(experiment_id))


@commands.command("rename")
@EXPERIMENT_ID
@click.argument("new_name")
def rename_experiment(experiment_id, new_name):
    """
    Renames an active experiment.
    Returns an error if the experiment is inactive.
    """
    store = _get_store()
    store.rename_experiment(experiment_id, new_name)
    print("Experiment with id %s has been renamed to '%s'." % (experiment_id, new_name))
