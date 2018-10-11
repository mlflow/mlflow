"""
The ``mlflow.azureml`` module provides an API for deploying MLflow models to Azure
Machine Learning.
"""
from __future__ import print_function


import os
import shutil
import tempfile

from azureml.core.image import ContainerImage
from azureml.core import Workspace

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.logging_utils import eprint
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.version import VERSION as mlflow_version


# def deploy(app_name, model_path, run_id=None, mlflow_home=None):
#     """
#     Deploy an MLflow model to Azure Machine Learning.
#
#     NOTE:
#
#         - This command must be called from a console launched from Azure Machine Learning Workbench.
#           Caller is reponsible for setting up Azure Machine Learning environment and accounts.
#
#         - Azure Machine Learning cannot handle any Conda environment. In particular the Python
#           version is fixed. If the model contains Conda environment and it has been trained outside
#           of Azure Machine Learning, the Conda environment might need to be edited to work with
#           Azure Machine Learning.
#
#     :param app_name: Name of the deployed application.
#     :param model_path: Local or MLflow-run-relative path to the model to be deployed.
#     :param run_id: MLflow run ID.
#     :param mlflow_home: Directory containing checkout of the MLflow GitHub project or
#                         current directory if not specified.
#     """
#     if run_id:
#         model_path = _get_model_log_dir(model_path, run_id)
#     model_path = os.path.abspath(model_path)
#     with TempDir(chdr=True, remove_on_exit=True):
#         exec_str = _export(app_name, model_path, mlflow_home=mlflow_home)
#         eprint("executing", '"{}"'.format(exec_str))
#         # Use os.system instead of subprocess due to the fact that currently all azureml commands
#         # have to be called within the same shell (launched from azureml workbench app by the user).
#         # We can change this once there is a python api (or general cli) available.
#         os.system(exec_str)
#


SCORE_SRC = """
import pandas as pd

from mlflow.pyfunc import load_pyfunc
from mlflow.utils import get_jsonable_obj


def init():
    global model
    model = load_pyfunc("{model_path}")


def run(s):
    input_df = pd.read_json(s, orient="records")
    return get_jsonable_obj(model.predict(input_df))

"""
def build_image(model_path, workspace, run_id=None):
    """
    :param model_path: The absolute path to MLflow model for which the image is being built
    :param workspace: The AzureML workspace in which to build the image. This is a 
                      `azureml.core.Workspace` object.
    :return: The name of the image that was created within the specified AzureML workspace.
    """
    if run_id is not None:
        model_path = _get_model_log_dir(model_name=model_path, run_id=run_id)

    image_name = "mlflow-{id}".format(id=_get_azureml_resource_unique_id())

    with TempDir() as tmp:
        tmp_model_path = tmp.path("model")
        tmp_model_subpath = _copy_file_or_tree(src=model_path, dst=tmp_model_path)
        tmp_model_path = os.path.join(tmp_model_path, tmp_model_subpath)

        # AzureML requires the container's execution script to be located
        # in the current working directory during image creation, so we
        # create the execution script as a temporary file in the current directory
        driver_file = tempfile.NamedTemporaryFile(dir=os.getcwd(), mode="w", 
                                                  prefix="driver", suffix=".py")
        driver_text = SCORE_SRC.format(
                model_path=("/var/azureml-app" + os.path.abspath(tmp_model_path)))
        driver_file.write(driver_text)

        conda_env_path = tmp.path("conda_env.yaml")
        _mlflow_conda_env(conda_env_path, 
                          additional_pip_deps=["mlflow=={mlflow_version}".format(
                              mlflow_version=mlflow_version)])

        image_configuration = ContainerImage.image_configuration(
                execution_script=driver_file.name,
                runtime="python",
                dependencies=[tmp_model_path],
                conda_file=conda_env_path)

        image = ContainerImage.create(workspace=workspace,
                                      name=image_name,
                                      image_config=image_configuration,
                                      models=[])
        image.wait_for_creation(show_output=True)
        return image_name


def export(output, model_path, run_id=None, mlflow_home=None):
    """
    Export an MLflow model with everything needed to deploy on Azure Machine Learning.
    Output includes sh script with command to deploy the generated model to Azure Machine Learning.

    NOTE:

        - This command does not need an Azure Machine Learning environment to run.

        - Azure Machine Learning cannot handle any Conda environment. In particular the Python
          version is fixed. If the model contains Conda environment and it has been trained outside
          of Azure Machine Learning, the Conda environment might need to be edited to work with
          Azure Machine Learning.

    :param output: Output folder where the model is going to be exported to.
    :param model_path: Local or MLflow run relative path to the model to be exported.
    :param run_id: MLflow run ID.
    :param mlflow_home: Directory containing checkout of the MLflow GitHub project or
                        current directory if not specified.
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


def _get_azureml_resource_unique_id():
    """
    :return: A unique identifier that can be appended to a user-readable resource name to avoid
             naming collisions.
    """
    import uuid
    import base64
    uuid_bytes = uuid.uuid4().bytes
    # Use base64 encoding to shorten the UUID length. Note that the replacement of the
    # unsupported '+' symbol maintains uniqueness because the UUID byte string is of a fixed,
    # 32-byte length
    uuid_b64 = base64.b64encode(uuid_bytes)
    if sys.version_info >= (3, 0):
        # In Python3, `uuid_b64` is a `bytes` object. It needs to be
        # converted to a string
        uuid_b64 = uuid_b64.decode("ascii")
    uuid_b64 = uuid_b64.rstrip('=\n').replace("/", "-").replace("+", "AB")
    return uuid_b64.lower()


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    workspace = Workspace.get("corey-azuresdk-test1")
    build_image(model_path=sys.argv[1], workspace=workspace, run_id=None)
