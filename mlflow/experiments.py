from __future__ import print_function

import os

import click
from tabulate import tabulate

from mlflow.data import is_uri
from mlflow.tracking import _get_store


@click.group("experiments")
def commands():
    """Manage experiments."""
    pass


@commands.command()
@click.argument("experiment_name")
def create(experiment_name):
    """
    Creates a new experiment in the configured tracking server.
    """
    store = _get_store()
    exp_id = store.create_experiment(experiment_name)
    print("Created experiment '%s' with id %d" % (experiment_name, exp_id))


@commands.command("list")
def list_experiments():
    """
    List all experiments in the configured tracking server.
    """
    store = _get_store()
    experiments = store.list_experiments()
    table = [[exp.experiment_id, exp.name, exp.artifact_location if is_uri(exp.artifact_location)
              else os.path.abspath(exp.artifact_location)] for exp in experiments]
    print(tabulate(sorted(table), headers=["Experiment Id", "Name", "Artifact Location"]))
