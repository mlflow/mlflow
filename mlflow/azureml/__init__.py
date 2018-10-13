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


def build_image(model_path, workspace, run_id=None, mlflow_home=None, description=None, tags={}):
    """
    :param model_path: The absolute path to MLflow model for which the image is being built
    :param workspace: The AzureML workspace in which to build the image. This is a 
                      `azureml.core.Workspace` object.
    :param mlflow_home: Path to a local copy of the MLflow GitHub repository. If specified, the 
                        image will install MLflow from this directory. Otherwise, it will install
                        MLflow from pip.
    :param tags: A collection of tags to give the Docker image when pushing to the Azure Container
                 Registry. These tags will be added to a set of default tags that include the model 
                 path, the model run id (if specified), and more. For more information, see 
                 https://docs.microsoft.com/en-us/python/api/azureml-core/
                 azureml.core.image.container.containerimageconfig.
    :param description: A string description to give the Docker image when pushing to the Azure
                        Container Registry. For more information, see 
                        https://docs.microsoft.com/en-us/python/api/azureml-core/
                        azureml.core.image.container.containerimageconfig.
    :return: An `azureml.core.image.ContainerImage` object containing metadata for the new image.
    """
    if run_id is not None:
        model_path = _get_model_log_dir(model_name=model_path, run_id=run_id)
    mlflow_pyfunc_conf = _load_pyfunc_conf(path=model_path)

    with TempDir() as tmp:
        tmp_model_path = tmp.path("model")
        model_path = os.path.join(
            tmp_model_path, _copy_file_or_tree(src=model_path, dst=tmp_model_path))
        file_dependencies = [model_path]

        execution_script_file = _create_execution_script(model_path=model_path)
        execution_script_path = os.path.basename(execution_script_file.name)

        if mlflow_home is not None:
            print("Copying the specified mlflow_home directory: {mlflow_home} to a temporary"
                  " location for container creation".format(mlflow_home=mlflow_home))
            tmp_mlflow_path = tmp.path("mlflow")
            mlflow_home = os.path.join(
                    tmp_mlflow_path, _copy_file_or_tree(src=mlflow_home, dst=tmp_mlflow_path))
            file_dependencies.append(mlflow_home)
        dockerfile_path = tmp.path("Dockerfile")
        _create_dockerfile(path=dockerfile_path, mlflow_path=mlflow_home)

        conda_env_path = None
        if pyfunc.ENV in model_pyfunc_conf:
            conda_env_path = os.path.join(model_path, model_pyfunc_conf[pyfunc.ENV])

        image_configuration = ContainerImage.image_configuration(
                execution_script=execution_script_path,
                runtime="python",
                docker_file=dockerfile_path,
                dependencies=file_dependencies,
                conda_file=conda_env_path,
                description=description,
                tags=tags,
        )
        image_name = "mlflow-{id}".format(id=_get_azureml_resource_unique_id())
        print("A new docker image will be built with the name: {image_name}".format(
            image_name=image_name))
        image = ContainerImage.create(workspace=workspace,
                                      name=image_name,
                                      image_config=image_configuration,
                                      models=[])
        image.wait_for_creation(show_output=True)
        return image


def _build_image_tags(model_path, run_id, model_pyfunc_conf, user_tags):
    image_tags = {
        "model_path": model_path,
    }
    if run_id is not None:
        image_tags["model_run_id"] = run_id
    if pyfunc.PY_VERSION in model_pyfunc_conf:
        image_tags["model_python_version"] = model_pyfunc_conf[pyfunc.PY_VERSION]



def _create_execution_script(model_path):
    # AzureML requires the container's execution script to be located
    # in the current working directory during image creation, so we
    # create the execution script as a temporary file in the current directory
    execution_script_file = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), mode="w", prefix="driver", suffix=".py")
    execution_script_text = SCORE_SRC.format(model_path=_get_container_path(model_path))
    execution_script_file.write(execution_script_text)
    execution_script_file.seek(0)
    return execution_script_file 


def _create_dockerfile(path, mlflow_path=None):
    if mlflow_path is not None:
        docker_cmd = "RUN pip install -e {mlflow_path}".format(
            mlflow_path=_get_container_path(mlflow_path))
    else:
        docker_cmd = "RUN pip install mlflow=={mlflow_version}".format(
            mlflow_version=mlflow_version)
    with open(path, "w") as f:
        f.write(docker_cmd)


def _get_container_path(host_dependency_path):
    if host_dependency_path.startswith("/"):
        host_dependency_path = host_dependency_path[1:]
    return os.path.join("/var/azureml-app", host_dependency_path)


def _load_pyfunc_conf(path):
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


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    if len(sys.argv) > 2:
        mlflow_home = sys.argv[2]
    else:
        mlflow_home = None
    workspace = Workspace.get("corey-azuresdk-east1")
    build_image(model_path=sys.argv[1], workspace=workspace, run_id=None, mlflow_home=mlflow_home)
