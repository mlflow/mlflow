from __future__ import print_function

import os
import shutil

import click

from mlflow import pyfunc

from mlflow.models import Model
from mlflow.tracking import _get_model_log_dir

from mlflow.utils.file_utils import TempDir


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
    """Deploy MLflow model to Azure ML.

    This command will export MLflow model into Azure ML compatible format and create a service
    models this model.

    NOTE: This command is to be called from correctly initialized Azure ML environment.
         At the moment this means it has to be run from console launched from Azure ML Workbench.
         Caller is reponsible for setting up Azure ML environment and accounts.

    NOTE: Azure ML can not handle any Conda environment. In particular python version seems to be
          fixed. If the model contains Conda environment and it has been trained outside of Azure
          ML, the Conda environment might need to be edited to work with Azure ML.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    model_path = os.path.abspath(model_path)
    with TempDir(chdr=True, remove_on_exit=True):
        exec_str = _export(app_name, model_path, "model")
        print("executing", '"{}"'.format(exec_str))
        # Use os.system instead of subprocess due to the fact that currently all azureml commands
        # have to be called within the same shell (launched from azureml workbench app by the user).
        # We can change this once there is a python api (or general cli) available.
        os.system(exec_str)


@commands.command("export")
@click.option("--output", "-o", default=None, help="Output directory.", required=True)
@click.option("--model-path", "-m", default=None,
              help="Path to the model. The path is relative to the run with the given run_id or "
                   "local filesystem path without run_id.", required=True)
@click.option("--run_id", "-r", default=None, help="Id of the MLflow run that contains the model.",
              required=False)
def export(output, model_path, run_id):
    """Export MLflow model as Azure ML compatible model ready to be deployed.

    Export MLflow model out with everything needed to deploy on Azure ML.
    Output includes sh script with command to deploy the generated model to Azure ML.
    The generated model has no dependency on MLflow.

    NOTE: This commnand does not need Azure ML environment to run.

    NOTE: Azure ML can not handle any Conda environment. In particular python version seems to be
          fixed. If the model contains Conda environment and it has been trained outside of Azure
          ML, the Conda environment might need to be edited.
    """
    output = os.path.abspath(output)
    if os.path.exists(output):
        raise Exception("output folder {} already exists".format(output))
    os.mkdir(output)
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    model_path = os.path.abspath(model_path)
    curr_dir = os.path.abspath(os.getcwd())
    os.chdir(output)
    try:
        exec_str = _export("$1", model_path, os.path.basename(output))
        with open("create_service.sh", "w") as f:
            f.write("\n".join(["#! /bin/sh", "cd {}".format(output), exec_str, ""]))
    finally:
        os.chdir(curr_dir)


def _export(app_name, model_path, dst_model_path):
    conf = _load_conf(model_path)
    score_py = "score.py"  # NOTE: azure ml requires the main module to be in the current directory
    loader_src = pyfunc.get_module_loader_src(model_path, dst_model_path)
    with open(score_py, "w") as f:
        f.write(SCORE_SRC.format(loader=loader_src))
    shutil.copytree(src=model_path, dst=dst_model_path)
    deps = ""
    env = "-c {}".format(os.path.join(dst_model_path, conf[pyfunc.ENV])) \
        if pyfunc.ENV in conf else ""
    cmd = "az ml service create realtime -n {name} " + \
          "--model-file {path} -f {score} {conda_env} {deps} -r python "
    return cmd.format(name=app_name, path=dst_model_path, score=score_py, conda_env=env, deps=deps)


def _load_conf(path):
    path = os.path.abspath(path)
    model = Model.load(os.path.join(path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise Exception("Currently only supports pyfunc format.")
    return model.flavors[pyfunc.FLAVOR_NAME]


SCORE_SRC = """
import pandas as pd
import json

{loader}

def init():
    global model
    model = load_pyfunc()


def run(s):    
    input_df = pd.read_json(s, orient="records")
    pred = model.predict(input_df)
    return json.dumps(pred.tolist())
"""
