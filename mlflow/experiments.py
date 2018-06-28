from __future__ import print_function

import os

import click
from tabulate import tabulate

from mlflow.store import file_store as store


@click.group("experiments")
def commands():
    """Manage experiments."""
    pass


@commands.command()
@click.option("--file-store", default=None,
              help="The root of the backing file store for experiment and run data "
                   "(default: ./mlruns).")
@click.option("--artifact-root", metavar="URI", default=None,
              help="Local or S3 URI to store artifacts in (default: inside file store).")
@click.argument("experiment_name")
def create(file_store, artifact_root, experiment_name):
    """
    Creates a new experiment in FileStore backend.
    """
    fs = store.FileStore(file_store, artifact_root)
    exp_id = fs.create_experiment(experiment_name)
    print("Created experiment '%s' with id %d" % (experiment_name, exp_id))


@commands.command("list")
@click.option("--file-store", default=None,
              help="The root of the backing file store for experiment and run data "
                   "(default: ./mlruns).")
def list_experiments(file_store):
    """
    List all experiment in FileStore backend.
    """
    fs = store.FileStore(file_store)
    experiments = fs.list_experiments()
    table = [[exp.experiment_id, exp.name, os.path.abspath(exp.artifact_location)]
             for exp in experiments]
    print(tabulate(sorted(table), headers=["Experiment Id", "Name", "Artifact Location"]))
