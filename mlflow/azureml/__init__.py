"""
The ``mlflow.azureml`` module provides an API for deploying MLflow models to Azure
Machine Learning.
"""
from __future__ import print_function

import sys
import os
import shutil
import tempfile
import logging

from distutils.version import StrictVersion

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils import PYTHON_VERSION, get_unique_resource_id
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree, _copy_project
from mlflow.version import VERSION as mlflow_version


_logger = logging.getLogger(__name__)


def build_image(model_path, workspace, run_id=None, image_name=None, model_name=None,
                mlflow_home=None, description=None, tags=None, synchronous=True):
    """
    Register an MLflow model with Azure ML and build an Azure ML ContainerImage for deployment.
    The resulting image can be deployed as a web service to Azure Container Instances (ACI) or
    Azure Kubernetes Service (AKS).

    The resulting Azure ML ContainerImage will contain a webserver that processes model queries.
    For information about the input data formats accepted by this webserver, see the
    :ref:`MLflow deployment tools documentation <azureml_deployment>`.

    :param model_path: The path to MLflow model for which the image will be built. If a run id
                       is specified, this is should be a run-relative path. Otherwise, it
                       should be a local path.
    :param run_id: MLflow run ID.
    :param image_name: The name to assign the Azure Container Image that will be created. If
                       unspecified, a unique image name will be generated.
    :param model_name: The name to assign the Azure Model will be created. If unspecified,
                       a unique model name will be generated.
    :param workspace: The AzureML workspace in which to build the image. This is a
                      `azureml.core.Workspace` object.
    :param mlflow_home: Path to a local copy of the MLflow GitHub repository. If specified, the
                        image will install MLflow from this directory. Otherwise, it will install
                        MLflow from pip.
    :param description: A string description to associate with the Azure Container Image and the
                        Azure Model that will be created. For more information, see
                        `<https://docs.microsoft.com/en-us/python/api/azureml-core/
                        azureml.core.image.container.containerimageconfig>`_ and
                        `<https://docs.microsoft.com/en-us/python/api/azureml-core/
                        azureml.core.model.model?view=azure-ml-py#register>`_.
    :param tags: A collection of tags, represented as a dictionary of string key-value pairs, to
                 associate with the Azure Container Image and the Azure Model that will be created.
                 These tags will be added to a set of default tags that include the model path,
                 the model run id (if specified), and more. For more information, see
                 `<https://docs.microsoft.com/en-us/python/api/azureml-core/
                 azureml.core.image.container.containerimageconfig>`_ and
                 `<https://docs.microsoft.com/en-us/python/api/azureml-core/
                 azureml.core.model.model?view=azure-ml-py#register>`_.
    :param synchronous: If `True`, this method will block until the image creation procedure
                        terminates before returning. If `False`, the method will return immediately,
                        but the returned image will not be available until the asynchronous
                        creation process completes. The `azureml.core.Image.wait_for_creation()`
                        function can be used to wait for the creation process to complete.
    :return: A tuple containing the following elements in order:
             - An `azureml.core.image.ContainerImage` object containing metadata for the new image.
             - An `azureml.core.model.Model` object containing metadata for the new model.

    >>> import mlflow.azureml
    >>> from azureml.core import Workspace
    >>> from azureml.core.webservice import AciWebservice, Webservice
    >>>
    >>> # Load or create an Azure ML Workspace
    >>> workspace_name = "<Name of your Azure ML workspace>"
    >>> subscription_id = "<Your Azure subscription ID>"
    >>> resource_group = "<Name of the Azure resource group in which to create Azure ML resources>"
    >>> location = "<Name of the Azure location (region) in which to create Azure ML resources>"
    >>> azure_workspace = Workspace.create(name=workspace_name,
    >>>                                    subscription_id=subscription_id,
    >>>                                    resource_group=resource_group,
    >>>                                    location=location,
    >>>                                    create_resource_group=True,
    >>>                                    exist_okay=True)
    >>>
    >>> # Build an Azure ML Container Image for an MLflow model
    >>> azure_image, azure_model = mlflow.azureml.build_image(
    >>>                                 model_path="<model_path>",
    >>>                                 workspace=azure_workspace,
    >>>                                 synchronous=True)
    >>> # If your image build failed, you can access build logs at the following URI:
    >>> print("Access the following URI for build logs: {}".format(azure_image.image_build_log_uri))
    >>>
    >>> # Deploy the image to Azure Container Instances (ACI) for real-time serving
    >>> webservice_deployment_config = AciWebservice.deploy_configuration()
    >>> webservice = Webservice.deploy_from_image(
    >>>                    image=azure_image, workspace=azure_workspace, name="<deployment-name>")
    >>> webservice.wait_for_deployment()
    """
    # The Azure ML SDK is only compatible with Python 3. However, the `mlflow.azureml` module should
    # still be accessible for import from Python 2. Therefore, we will only import from the SDK
    # upon method invocation.
    # pylint: disable=import-error
    from azureml.core.image import ContainerImage
    from azureml.core.model import Model as AzureModel

    if run_id is not None:
        absolute_model_path = _get_model_log_dir(model_name=model_path, run_id=run_id)
    else:
        absolute_model_path = os.path.abspath(model_path)

    model_pyfunc_conf = _load_pyfunc_conf(model_path=absolute_model_path)
    model_python_version = model_pyfunc_conf.get(pyfunc.PY_VERSION, None)
    if model_python_version is not None and\
            StrictVersion(model_python_version) < StrictVersion("3.0.0"):
        raise MlflowException(
                message=("Azure ML can only deploy models trained in Python 3 or above! Please see"
                         " the following MLflow GitHub issue for a thorough explanation of this"
                         " limitation and a workaround to enable support for deploying models"
                         " trained in Python 2: https://github.com/mlflow/mlflow/issues/668"),
                error_code=INVALID_PARAMETER_VALUE)

    tags = _build_tags(relative_model_path=model_path, run_id=run_id,
                       model_python_version=model_python_version, user_tags=tags)

    if image_name is None:
        image_name = _get_mlflow_azure_resource_name()
    if model_name is None:
        model_name = _get_mlflow_azure_resource_name()

    with TempDir(chdr=True) as tmp:
        model_directory_path = tmp.path("model")
        tmp_model_path = os.path.join(
            model_directory_path,
            _copy_file_or_tree(src=absolute_model_path, dst=model_directory_path))

        registered_model = AzureModel.register(workspace=workspace, model_path=tmp_model_path,
                                               model_name=model_name, tags=tags,
                                               description=description)
        _logger.info("Registered an Azure Model with name: `%s` and version: `%s`",
                     registered_model.name, registered_model.version)

        # Create an execution script (entry point) for the image's model server. Azure ML requires
        # the container's execution script to be located in the current working directory during
        # image creation, so we create the execution script as a temporary file in the current
        # working directory.
        execution_script_path = tmp.path("execution_script.py")
        _create_execution_script(output_path=execution_script_path, azure_model=registered_model)
        # Azure ML copies the execution script into the image's application root directory by
        # prepending "/var/azureml-app" to the specified script path. The script is then executed
        # by referencing its path relative to the "/var/azureml-app" directory. Unfortunately,
        # if the script path is an absolute path, Azure ML attempts to reference it directly,
        # resulting in a failure. To circumvent this problem, we provide Azure ML with the relative
        # script path. Because the execution script was created in the current working directory,
        # this relative path is the script path's base name.
        execution_script_path = os.path.basename(execution_script_path)

        if mlflow_home is not None:
            _logger.info(
                "Copying the specified mlflow_home directory: `%s` to a temporary location for"
                " container creation",
                mlflow_home)
            mlflow_home = os.path.join(tmp.path(),
                                       _copy_project(src_path=mlflow_home, dst_path=tmp.path()))
            image_file_dependencies = [mlflow_home]
        else:
            image_file_dependencies = None
        dockerfile_path = tmp.path("Dockerfile")
        _create_dockerfile(output_path=dockerfile_path, mlflow_path=mlflow_home)

        conda_env_path = None
        if pyfunc.ENV in model_pyfunc_conf:
            conda_env_path = os.path.join(tmp_model_path, model_pyfunc_conf[pyfunc.ENV])

        image_configuration = ContainerImage.image_configuration(
                execution_script=execution_script_path,
                runtime="python",
                docker_file=dockerfile_path,
                dependencies=image_file_dependencies,
                conda_file=conda_env_path,
                description=description,
                tags=tags,
        )
        image = ContainerImage.create(workspace=workspace,
                                      name=image_name,
                                      image_config=image_configuration,
                                      models=[registered_model])
        _logger.info("Building an Azure Container Image with name: `%s` and version: `%s`",
                     image.name, image.version)
        if synchronous:
            image.wait_for_creation(show_output=True)
        return image, registered_model


def _build_tags(relative_model_path, run_id, model_python_version=None, user_tags=None):
    """
    :param model_path: The path to MLflow model for which the image is being built. If a run id
                       is specified, this is a run-relative path. Otherwise, it is a local path.
    :param run_id: MLflow run ID.
    :param model_pyfunc_conf: The configuration for the `python_function` flavor within the
                              specified model's "MLmodel" configuration.
    :param user_tags: A collection of user-specified tags to append to the set of default tags.
    """
    tags = dict(user_tags) if user_tags is not None else {}
    tags["model_path"] = relative_model_path if run_id is not None\
        else os.path.abspath(relative_model_path)
    if run_id is not None:
        tags["run_id"] = run_id
    if model_python_version is not None:
        tags["python_version"] = model_python_version
    return tags


def _create_execution_script(output_path, azure_model):
    """
    Creates an Azure-compatibele execution script (entry point) for a model server backed by
    the specified model. This script is created as a temporary file in the current working
    directory.

    :param output_path: The path where the execution script will be written.
    :param azure_model: The Azure Model that the execution script will load for inference.
    :return: A reference to the temporary file containing the execution script.
    """
    execution_script_text = SCORE_SRC.format(
            model_name=azure_model.name, model_version=azure_model.version)

    with open(output_path, "w") as f:
        f.write(execution_script_text)


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
    docker_cmds = ["RUN pip install azureml-sdk"]

    if mlflow_path is not None:
        mlflow_install_cmd = "RUN pip install -e {mlflow_path}".format(
            mlflow_path=_get_container_path(mlflow_path))
    elif not mlflow_version.endswith("dev"):
        mlflow_install_cmd = "RUN pip install mlflow=={mlflow_version}".format(
            mlflow_version=mlflow_version)
    else:
        raise MlflowException(
                "You are running a 'dev' version of MLflow: `{mlflow_version}` that cannot be"
                " installed from pip. In order to build a container image, either specify the"
                " path to a local copy of the MLflow GitHub repository using the `mlflow_home`"
                " parameter or install a release version of MLflow from pip".format(
                    mlflow_version=mlflow_version))
    docker_cmds.append(mlflow_install_cmd)

    with open(output_path, "w") as f:
        f.write("\n".join(docker_cmds))


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
                error_code=INVALID_PARAMETER_VALUE)
    return model.flavors[pyfunc.FLAVOR_NAME]


def _get_mlflow_azure_resource_name():
    """
    :return: A unique name for an Azure resource indicating that the resource was created by
             MLflow
    """
    azureml_max_resource_length = 32
    resource_prefix = "mlflow-"
    unique_id = get_unique_resource_id(
            max_length=(azureml_max_resource_length - len(resource_prefix)))
    return resource_prefix + unique_id


SCORE_SRC = """
import pandas as pd

from azureml.core.model import Model
from mlflow.pyfunc import load_pyfunc
from mlflow.pyfunc.scoring_server import parse_json_input
from mlflow.utils import get_jsonable_obj


def init():
    global model
    model_path = Model.get_model_path(model_name="{model_name}", version={model_version})
    model = load_pyfunc(model_path)


def run(json_input):
    input_df = parse_json_input(json_input=json_input, orient="split")
    return get_jsonable_obj(model.predict(input_df), pandas_orient="records")

"""
