"""
Internal CLI commands used by MLflow APIs
"""
from __future__ import print_function

import click
import mlflow


@click.group("internals")
def commands():
    """
    CLI commands used internally by MLflow APIs. Not intended to be run as part of normal MLflow
    usage.
    """
    pass


@commands.command()
@click.argument("entry_point_command")
def _run_internal(entry_point_command):
    """
    Internally-used CLI command for running entry points locally.
    """
    mlflow.projects._run_entry_point_command(entry_point_command)
