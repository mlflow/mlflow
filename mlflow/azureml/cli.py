"""
CLI for azureml module.
"""
from __future__ import print_function

import os

import click

import mlflow
import mlflow.azureml

from mlflow.utils import cli_args


@click.group("azureml")
def commands():
    """
    Serve models on Azure ML.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("deploy")
@click.option("--app-name", "-n", default=None,
              help="The application name under which should this model be deployed. "
                   "Translates to service name on Azure ML", required=True)
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@cli_args.MLFLOW_HOME
def deploy(app_name, model_path, run_id, mlflow_home):
    """Deploy MLflow model to Azure ML.

       NOTE: This command must be run from console launched from Azure ML Workbench.
            Caller is reponsible for setting up Azure ML environment and accounts.

       NOTE: Azure ML cannot handle any Conda environment. In particular the Python version is
             fixed. If the model contains Conda environment and it has been trained outside of Azure
             ML, the Conda environment might need to be edited to work with Azure ML.
    """
    mlflow.azureml.deploy(app_name=app_name, model_path=model_path, run_id=run_id,
                          mlflow_home=os.path.abspath(mlflow_home) if mlflow_home else None)


@commands.command("export")
@click.option("--output", "-o", default=None, help="Output directory.", required=True)
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@cli_args.MLFLOW_HOME
def export(output, model_path, run_id, mlflow_home):
    """Export MLflow model as Azure ML compatible model ready to be deployed.

    Export MLflow model with everything needed to deploy on Azure ML.
    Output includes sh script with command to deploy the generated model to Azure ML.

    NOTE: This command does not need Azure ML environment to run.

    NOTE: Azure ML can not handle any Conda environment. If the model contains Conda environment
    and it has been trained outside of Azure ML, the Conda environment might need to be edited.
    """
    mlflow.azureml.export(output=output, model_path=model_path, run_id=run_id,
                          mlflow_home=os.path.abspath(mlflow_home) if mlflow_home else None)
