"""
The ``mlflow.azureml`` module provides an API for deploying MLflow models to Azure
Machine Learning.
"""
from __future__ import print_function

import sys
import os
import shutil
import tempfile

from azureml.core.image import ContainerImage
from azureml.core import Workspace

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException 
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST 
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.logging_utils import eprint
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.version import VERSION as mlflow_version


def build_container_image(model_path, workspace, run_id=None, mlflow_home=None, description=None, 
                          tags={}, synchronous=True):
    """
    Builds an Azure ML ContainerImage for the specified model. This image can be deployed as a
    web service to Azure Container Instances (ACI) or Azure Kubernetes Service (AKS).

    :param model_path: The path to MLflow model for which the image is being built. If a run id
                       is specified, this is a run-relative path. Otherwise, it is a local path.
    :param run_id: MLflow run ID.
    :param workspace: The AzureML workspace in which to build the image. This is a 
                      `azureml.core.Workspace` object.
    :param mlflow_home: Path to a local copy of the MLflow GitHub repository. If specified, the 
                        image will install MLflow from this directory. Otherwise, it will install
                        MLflow from pip.
    :param description: A string description to give the Docker image when pushing to the Azure
                        Container Registry. For more information, see 
                        https://docs.microsoft.com/en-us/python/api/azureml-core/
                        azureml.core.image.container.containerimageconfig.
    :param tags: A collection of tags to give the Docker image when pushing to the Azure Container
                 Registry. These tags will be added to a set of default tags that include the model 
                 path, the model run id (if specified), and more. For more information, see 
                 https://docs.microsoft.com/en-us/python/api/azureml-core/
                 azureml.core.image.container.containerimageconfig.
    :param synchronous: If `True`, this method will block until the image creation procedure
                        terminates before returning. If `False`, the method will return immediately,
                        but the returned image will not be available until the asynchronous
                        creation process completes. The `azureml.core.Image.wait_for_creation()`
                        function can be used to wait for the creation process to complete.
    :return: An `azureml.core.image.ContainerImage` object containing metadata for the new image.
    """
    if run_id is not None:
        model_path = _get_model_log_dir(model_name=model_path, run_id=run_id)
    model_pyfunc_conf = _load_pyfunc_conf(model_path=model_path)
    image_tags = _build_image_tags(model_path=model_path, run_id=run_id, 
                                   model_pyfunc_conf=model_pyfunc_conf, 
                                   user_tags=tags)

    with TempDir() as tmp:
        tmp_model_path = tmp.path("model")
        model_path = os.path.join(
            tmp_model_path, _copy_file_or_tree(src=model_path, dst=tmp_model_path))
        file_dependencies = [model_path]

        # Create an execution script (entry point) for the image's model server in the
        # current working directory
        execution_script_file = _create_execution_script(model_path=model_path)
        # Azure ML copies the execution script into the image's application root directory by
        # prepending "/var/azureml-app" to the specified script path. The script is then executed
        # by referencing its path relative to the "/var/azureml-app" directory. Unfortunately,
        # if the script path is an **absolute path**, Azure ML attempts to reference it directly,
        # resulting in a failure. To circumvent this problem, we provide Azure ML with the relative 
        # script path. Because the execution script was created in the current working directory, 
        # this relative path is the script path's base name.
        execution_script_path = os.path.basename(execution_script_file.name)

        if mlflow_home is not None:
            eprint("Copying the specified mlflow_home directory: {mlflow_home} to a temporary"
                  " location for container creation".format(mlflow_home=mlflow_home))
            tmp_mlflow_path = tmp.path("mlflow")
            mlflow_home = os.path.join(
                    tmp_mlflow_path, _copy_file_or_tree(src=mlflow_home, dst=tmp_mlflow_path))
            file_dependencies.append(mlflow_home)
        dockerfile_path = tmp.path("Dockerfile")
        _create_dockerfile(output_path=dockerfile_path, mlflow_path=mlflow_home)

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
        image_name = "mlflow-{uid}".format(uid=_get_azureml_resource_unique_id())
        eprint("A new docker image will be built with the name: {image_name}".format(
            image_name=image_name))
        image = ContainerImage.create(workspace=workspace,
                                      name=image_name,
                                      image_config=image_configuration,
                                      models=[])
        if synchronous:
            image.wait_for_creation(show_output=True)
        return image


def _build_image_tags(model_path, run_id, model_pyfunc_conf, user_tags):
    """
    :param model_path: The path to MLflow model for which the image is being built. If a run id
                       is specified, this is a run-relative path. Otherwise, it is a local path.
    :param run_id: MLflow run ID.
    :param model_pyfunc_conf: The configuration for the `python_function` flavor within the
                              specified model's "MLmodel" configuration.
    :param user_tags: A collection of user-specified tags to add to the image, in addition to
                      the default set of tags.
    """
    image_tags = dict(user_tags)
    image_tags["model_path"] = model_path if run_id is not None else os.path.abspath(model_path)
    if run_id is not None:
        image_tags["model_run_id"] = run_id
    if pyfunc.PY_VERSION in model_pyfunc_conf:
        image_tags["model_python_version"] = model_pyfunc_conf[pyfunc.PY_VERSION]
    return image_tags


def _create_execution_script(model_path):
    """
    Creates an Azure-compatibele execution script (entry point) for a model server backed by
    the specified model. This script is created as a temporary file in the current working 
    directory.

    :param model_path: The absolute path to the model for which to create an execution script.
    :return: A reference to the temporary file containing the execution script.
    """
    # Azure ML requires the container's execution script to be located
    # in the current working directory during image creation, so we
    # create the execution script as a temporary file in the current directory
    execution_script_file = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), mode="w", prefix="driver", suffix=".py")
    execution_script_text = SCORE_SRC.format(model_path=_get_container_path(model_path))
    execution_script_file.write(execution_script_text)
    execution_script_file.seek(0)
    return execution_script_file 


def _create_dockerfile(output_path, mlflow_path=None):
    """
    Creates a Dockerfile containing additional Docker build steps to execute
    when building the Azure container image. These build steps perform the following tasks:

    - Install MLflow

    :param output_path: The path where the Dockerfile will be written.
    :param mlflow_path: Path to a local copy of the MLflow GitHub repository. If specified, the
                        Dockerfile command for MLflow installation will install MLflow from this
                        directory. Otherwise, it will install MLflow from pip.
    """
    if mlflow_path is not None:
        docker_cmd = "RUN pip install -e {mlflow_path}".format(
            mlflow_path=_get_container_path(mlflow_path))
    else:
        docker_cmd = "RUN pip install mlflow=={mlflow_version}".format(
            mlflow_version=mlflow_version)
    with open(output_path, "w") as f:
        f.write(docker_cmd)


def _get_container_path(local_path):
    """
    Given a local path to a resource, obtains the path at which this resource will exist 
    when it is copied into the Azure ML container image.
    """
    if local_path.startswith("/"):
        local_path = local_path[1:]
    return os.path.join("/var/azureml-app", local_path)


def _load_pyfunc_conf(model_path):
    """
    Loads the `python_function` flavor configuration for the specified model or throws an exception 
    if the model does not contain the `python_function` flavor.

    :param model_path: The absolute path to the model.
    :return: The model's `python_function` flavor configuration.
    """
    model_path = os.path.abspath(model_path)
    model = Model.load(os.path.join(model_path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise MlflowException(
                message=("The specified model does not contain the `python_function` flavor. This "
                         " flavor is required for model deployment required for model deployment."),
                         error_code=RESOURCE_DOES_NOT_EXIST)
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
