from __future__ import print_function


import os
import shutil

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking import _get_model_log_dir
from mlflow.utils.logging_utils import eprint
from mlflow.utils.file_utils import TempDir
from mlflow.version import VERSION as mlflow_version


def deploy(app_name, model_path, run_id, mlflow_home):
    """
    Deploy MLflow model to Azure ML.

    NOTE:

        - This command must be called from a console launched from Azure ML Workbench. Caller is
          reponsible for setting up Azure ML environment and accounts.

        - Azure ML can not handle any Conda environment. In particular the Python version is fixed.
          If the model contains Conda environment and it has been trained outside of Azure ML, the
          Conda environment might need to be edited to work with Azure ML.

    :param app_name: Name of the deployed application.
    :param model_path: Local or MLflow-run-relative path to the model to be exported.
    :param run_id: If provided, ``run_id`` is used to retrieve the model logged with MLflow.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    model_path = os.path.abspath(model_path)
    with TempDir(chdr=True, remove_on_exit=True):
        exec_str = _export(app_name, model_path, mlflow_home=mlflow_home)
        eprint("executing", '"{}"'.format(exec_str))
        # Use os.system instead of subprocess due to the fact that currently all azureml commands
        # have to be called within the same shell (launched from azureml workbench app by the user).
        # We can change this once there is a python api (or general cli) available.
        os.system(exec_str)


def export(output, model_path, run_id, mlflow_home):
    """
    Export MLflow model as Azure ML compatible model ready to be deployed.

    Export MLflow model with everything needed to deploy on Azure ML.
    Output includes sh script with command to deploy the generated model to Azure ML.

    NOTE:

        - This command does not need an Azure ML environment to run.

        - Azure ML can not handle any Conda environment. If the model contains Conda environment
          and it has been trained outside of Azure ML, the Conda environment might need to be
          edited.

    :param output: Output folder where the model is going to be exported to.
    :param model_path: Local or MLflow run relative path to the model to be exported.
    :param run_id: If provided, ``run_id`` is used to retrieve model logged with MLflow.
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
        exec_str = _export("$1", model_path, mlflow_home=mlflow_home)
        with open("create_service.sh", "w") as f:
            f.write("\n".join(["#! /bin/sh", "cd {}".format(output), exec_str, ""]))
    finally:
        os.chdir(curr_dir)


def _export(app_name, model_path, mlflow_home):
    conf = _load_conf(model_path)
    score_py = "score.py"  # NOTE: azure ml requires the main module to be in the current directory

    with open(score_py, "w") as f:
        f.write(SCORE_SRC)

    deps = ""
    mlflow_dep = "mlflow=={}".format(mlflow_version)

    if mlflow_home:
        eprint("MLFLOW_HOME =", mlflow_home)
        # copy current version of mlflow
        mlflow_dir = mlflow.utils.file_utils._copy_project(src_path=mlflow_home, dst_path="./")
        deps = "-d {}".format(mlflow_dir)
        mlflow_dep = "-e /var/azureml-app/{}".format(mlflow_dir)

    with open("requirements.txt", "w") as f:
        f.write(mlflow_dep + "\n")

    shutil.copytree(src=model_path, dst="model")

    env = "-c {}".format(os.path.join("model", conf[pyfunc.ENV])) \
        if pyfunc.ENV in conf else ""
    cmd = "az ml service create realtime -n {name} " + \
          "--model-file model -f score.py {conda_env} {deps} -r python -p requirements.txt"
    return cmd.format(name=app_name, conda_env=env, deps=deps)


def _load_conf(path):
    path = os.path.abspath(path)
    model = Model.load(os.path.join(path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise Exception("Supports only pyfunc format.")
    return model.flavors[pyfunc.FLAVOR_NAME]


SCORE_SRC = """
import pandas as pd

from mlflow.pyfunc import load_pyfunc
from mlflow.utils import get_jsonable_obj


def init():
    global model
    model = load_pyfunc("model")


def run(s):
    input_df = pd.read_json(s, orient="records")
    return get_jsonable_obj(model.predict(input_df))

"""
