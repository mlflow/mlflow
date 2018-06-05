from __future__ import print_function

import os

import click
from tabulate import tabulate

from mlflow.store import file_store


@click.group("experiments")
def commands():
    """Tracking APIs."""
    pass


@commands.command()
@click.option("--file-store-path", default=None,
              help="The root of the backing file store for experiment and run data. Defaults to %s."
                   % file_store._default_root_dir())
@click.argument("experiment_name")
def create(file_store_path, experiment_name):
    """
    Creates a new experiment in FileStore backend.
    """
    fs = file_store.FileStore(file_store_path)
    exp_id = fs.create_experiment(experiment_name)
    print("Created experiment '%s' with id '%d'" % (experiment_name, exp_id))


@commands.command("list")
@click.option("--file-store-path", default=None,
              help="The root of the backing file store for experiment and run data. Defaults to %s."
                   % file_store._default_root_dir())
def list_experiments(file_store_path):
    """
    List all experiment in FileStore backend.
    """
    fs = file_store.FileStore(file_store_path)
    experiments = fs.list_experiments()
    table = [[exp.experiment_id, exp.name, os.path.abspath(exp.artifact_location)]
             for exp in experiments]
    print(tabulate(sorted(table), headers=["Experiment Id", "Name", "Artifact Location"]))
