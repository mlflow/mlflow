"""
CLI for azureml module.
"""
import json

import click

import mlflow.azureml
from mlflow.utils import cli_args
from mlflow.utils.annotations import deprecated


@click.group("azureml")
def commands():
    """
    Serve models on Azure ML. **These commands require that MLflow be installed with Python 3.**

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command(
    short_help=(
        "Register an MLflow model with Azure ML and build an Azure ML"
        " ContainerImage for deployment."
    )
)
@cli_args.MODEL_URI
@click.option(
    "--workspace-name",
    "-w",
    required=True,
    help="The name of the Azure Workspace in which to build the image.",
)
@click.option(
    "--subscription-id",
    "-s",
    default=None,
    help=("The subscription id associated with the Azure Workspace in which to build" " the image"),
)
@click.option(
    "--image-name",
    "-i",
    default=None,
    help=(
        "The name to assign the Azure Container Image that is created. If unspecified,"
        " a unique image name will be generated."
    ),
)
@click.option(
    "--model-name",
    "-n",
    default=None,
    help=(
        "The name to assign the Azure Model that is created. If unspecified,"
        " a unique image name will be generated."
    ),
)
@cli_args.MLFLOW_HOME
@click.option(
    "--description",
    "-d",
    default=None,
    help=(
        "A string description to associate with the Azure Container Image and the"
        " Azure Model that are created."
    ),
)
@click.option(
    "--tags",
    "-t",
    default=None,
    help=(
        "A collection of tags, represented as a JSON-formatted dictionary of string"
        " key-value pairs, to associate with the Azure Container Image and the Azure"
        " Model that are created. These tags are added to a set of default tags"
        " that include the model path, the model run id (if specified), and more."
    ),
)
@deprecated("the azureml deployment plugin, https://aka.ms/aml-mlflow-deploy", since="1.19.0")
def build_image(
    model_uri,
    workspace_name,
    subscription_id,
    image_name,
    model_name,
    mlflow_home,
    description,
    tags,
):
    """
    Register an MLflow model with Azure ML and build an Azure ML ContainerImage for deployment.
    The resulting image can be deployed as a web service to Azure Container Instances (ACI) or
    Azure Kubernetes Service (AKS).

    The resulting Azure ML ContainerImage will contain a webserver that processes model queries.
    For information about the input data formats accepted by this webserver, see the following
    documentation: https://www.mlflow.org/docs/latest/models.html#azureml-deployment.
    """
    # The Azure ML SDK is only compatible with Python 3. However, this CLI should still be
    # accessible for inspection rom Python 2. Therefore, we will only import from the SDK
    # upon command invocation.
    # pylint: disable=import-error
    from azureml.core import Workspace

    workspace = Workspace.get(name=workspace_name, subscription_id=subscription_id)
    if tags is not None:
        tags = json.loads(tags)
    mlflow.azureml.build_image(
        model_uri=model_uri,
        workspace=workspace,
        image_name=image_name,
        model_name=model_name,
        mlflow_home=mlflow_home,
        description=description,
        tags=tags,
        synchronous=True,
    )
