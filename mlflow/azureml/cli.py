"""
CLI for azureml module.
"""
from __future__ import print_function

import click

import mlflow
import mlflow.azureml


@click.group("azureml")
def commands():
    """Serve models on Azure ML."""
    pass


@commands.command("deploy")
@click.option("--app-name", "-n", default=None,
              help="The application name under which should this model be deployed. "
                   "Translates to service name on Azure ML", required=True)
@click.option("--model-path", "-m", default=None,
              help="Path to the model. The path is relative to the run with the given run_id or "
                   "local filesystem path without run_id.", required=True)
@click.option("--run_id", "-r", default=None, help="Id of the MLflow run that contains the model.",
              required=False)
def deploy(app_name, model_path, run_id):
    mlflow.azureml.deploy(app_name=app_name, model_path=model_path, run_id=run_id)


@commands.command("export")
@click.option("--output", "-o", default=None, help="Output directory.", required=True)
@click.option("--model-path", "-m", default=None,
              help="Path to the model. The path is relative to the run with the given run_id or "
                   "local filesystem path without run_id.", required=True)
@click.option("--run_id", "-r", default=None, help="Id of the MLflow run that contains the model.",
              required=False)
def export(output, model_path, run_id):
    mlflow.azureml.export(output=output, model_path=model_path, run_id=run_id)
