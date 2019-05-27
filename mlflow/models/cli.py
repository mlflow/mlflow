import logging
import click
import posixpath

import mlflow.sagemaker
from mlflow.models import Model, docker_utils
from mlflow.models.flavor_backend_registry import get_flavor_backend,\
    get_flavor_backend_for_build_image
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
from mlflow.utils import cli_args
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@click.group("models")
def commands():
    """
    Deploy MLflow models locally.

    To deploy a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("serve")
@cli_args.MODEL_URI
@cli_args.PORT
@cli_args.HOST
@cli_args.WORKERS
@cli_args.NO_CONDA
@cli_args.INSTALL_MLFLOW
def serve(model_uri, port, host, workers, no_conda=False, install_mlflow=False):
    """
    Serve a model saved with MLflow by launching a webserver on the specified host and port. For
    information about the input data formats accepted by the webserver, see the following
    documentation: https://www.mlflow.org/docs/latest/models.html#model-deployment.
    """
    return _get_flavor_backend(model_uri,
                               no_conda=no_conda,
                               workers=workers,
                               install_mlflow=install_mlflow).serve(model_uri=model_uri, port=port,
                                                                    host=host)


@commands.command("predict")
@cli_args.MODEL_URI
@click.option("--input-path", "-i", default=None,
              help="CSV containing pandas DataFrame to predict against.")
@click.option("--output-path", "-o", default=None,
              help="File to output results to as json file. If not provided, output to stdout.")
@click.option("--content-type", "-t", default="json",
              help="Content type of the input file. Can be one of {'json', 'csv'}.")
@click.option("--json-format", "-j", default="split",
              help="Only applies if the content type is 'json'. Specify how the data is encoded.  "
                   "Can be one of {'split', 'records'} mirroring the behavior of Pandas orient "
                   "attribute. The default is 'split' which expects dict like data: "
                   "{'index' -> [index], 'columns' -> [columns], 'data' -> [values]}, "
                   "where index  is optional. For more information see "
                   "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json"
                   ".html")
@cli_args.NO_CONDA
@cli_args.INSTALL_MLFLOW
def predict(model_uri, input_path, output_path, content_type, json_format, no_conda,
            install_mlflow):
    """
    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#model-deployment.
    """
    if content_type == "json" and json_format not in ("split", "records"):
        raise Exception("Unsupported json format '{}'.".format(json_format))
    return _get_flavor_backend(model_uri, no_conda=no_conda,
                               install_mlflow=install_mlflow).predict(model_uri=model_uri,
                                                                      input_path=input_path,
                                                                      output_path=output_path,
                                                                      content_type=content_type,
                                                                      json_format=json_format)


@commands.command("build-docker")
@experimental
@cli_args.MODEL_URI
@click.option("--name", "-n", default="mlflow-pyfunc-servable",
              help="Name to use for built image")
@click.option("--flavor", "-f", default=None,
              help=("The name of the flavor to use for local serving. Must be one of the following:"
                    " {supported_flavors}. If unspecified, a flavor will be automatically selected"
                    " from the model's available flavors.".format(
                        supported_flavors=docker_utils.SUPPORTED_DEPLOYMENT_FLAVORS)))
def build_docker(model_uri, name, flavor):
    """
    [EXPERIMENTAL] Builds a Docker image containing the MLflow model at the specified URI.
    The image's entry point serves the model with default settings (i.e. as an endpoint exposed
    at port 8080 within the container). Note that the model is assumed to have the pyfunc flavor -
    see https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html for more information.

    This command is experimental and does not guarantee that the arguments nor format of the
    Docker container will remain the same.
    """
    _get_flavor_backend_for_image_build(model_uri, flavor).build_image(
        model_uri, name, mlflow_home=".")


def _get_flavor_backend_for_image_build(model_uri, flavor, **kwargs):
    with TempDir() as tmp:
        local_path = _download_artifact_from_uri(posixpath.join(model_uri, "MLmodel"),
                                                 output_path=tmp.path())
        model = Model.load(local_path)
    flavor_name, flavor_backend = get_flavor_backend_for_build_image(model, flavor, **kwargs)
    if flavor_backend is None:
        raise Exception("No suitable flavor backend was found for the model.")
    _logger.info("Selected backend for flavor '%s'", flavor_name)
    return flavor_backend


def _get_flavor_backend(model_uri, **kwargs):
    with TempDir() as tmp:
        local_path = _download_artifact_from_uri(posixpath.join(model_uri, "MLmodel"),
                                                 output_path=tmp.path())
        model = Model.load(local_path)
    flavor_name, flavor_backend = get_flavor_backend(model, **kwargs)
    if flavor_backend is None:
        raise Exception("No suitable flavor backend was found for the model.")
    _logger.info("Selected backend for flavor '%s'", flavor_name)
    return flavor_backend
