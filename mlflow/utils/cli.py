from __future__ import print_function

import os
import json

import click

import mlflow
from mlflow.utils import cli_args
import mlflow.utils.environment


@click.group("utils")
def commands():
    """MLflow utilities."""
    pass


@click.option("--force", default=False,
              help="Remove the matching environments without asking if this option is set.")

@click.option("--env-file", "-e", metavar="PATH", default=None,
              help="Remove cached conda environment only for the given conda yaml file if set.")
@commands.command("remove-cached-environments")
def remove_cached_environments(project_path, force):
    """
    Remove cached Conda environments.

    When MLflow executes a project with associated conda environment file it will create a new conda
    environment defined by this file. This environment is cached based on the content of the conda
    environment yaml file. This means it is only recreated if the project dependencies change.

    There are two reasons why you might need to remove these files:

    1. To save disk space. The environments are never removed and can consume significant amount of
       disk space.
    2. To force recreation of the environment in cases when the caching mechanism failed. This
       happens e.g. when the conda file includes a dependency without specified version.
    """
    mlflow.utils.environment._clear_conda_env_cache(remove_envs=[project_path], force=force)



@commands.command("list-cached-environments")
def list_cached_environments():
    """
    List cached Conda environments.

    When MLflow executes a project with associated conda environment file it will create a new conda
    environment defined by this file. This environment is cached based on the content of the conda
    environment yaml file. This means it is only recreated if the project dependencies change.

    This command lists the Conda environments created by MLflow.
    """
    mlflow.utils.logging_utils.eprint(
        "\n".join(mlflow.utils.environment._get_mlflow_environments()))
