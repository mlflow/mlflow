
import logging
import click
import posixpath

from mlflow.models import Model
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
from mlflow.utils import cli_args

_logger = logging.getLogger(__name__)


@click.group("models")
def commands():
    """
    Serve MLflow models locally.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("serve")
@cli_args.MODEL_URI
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--host", "-h", default="127.0.0.1", help="Server host. [default: 127.0.0.1]")
@cli_args.NO_CONDA
def serve(model_uri, port, host, no_conda=False):
    """
    Serve a model saved with MLflow by launching a webserver on the specified
    host and port. For information about the input data formats accepted by the webserver,
    see the following documentation:
    TODO: fix the link below
    https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment.
    """
    return _get_flavor_backend(model_uri, no_conda).serve(model_uri=model_uri, port=port, host=host,
                                                no_conda=no_conda)


@commands.command("predict")
@cli_args.MODEL_URI
@click.option("--input-path", "-i", default=None,
              help="CSV containing pandas DataFrame to predict against.")
@click.option("--output-path", "-o", default=None,
              help="File to output results to as CSV file. If not provided, output to stdout.")
@click.option("--content-type", "-t", default="json",
              help="Content type of the input file. Can be one of {'json', 'csv'}.")
@cli_args.NO_CONDA
def predict(model_uri, input_path, output_path, content_type, no_conda):
    """
    Load a pandas DataFrame and runs a python_function model saved with MLflow against it.
    Return the prediction results as a CSV-formatted pandas DataFrame.
    """
    return _get_flavor_backend(model_uri, no_conda).predict(model_uri=model_uri,
                                                            input_path=input_path,
                                                            output_path=output_path,
                                                            content_type=content_type,
                                                            no_conda=no_conda)


def _get_flavor_backend(model_uri, no_conda):
    with TempDir() as tmp:
        local_path = _download_artifact_from_uri(posixpath.join(model_uri, "MLmodel"),
                                                 output_path=tmp.path())
        model = Model.load(local_path)
    flavor_backend = get_flavor_backend(model, no_conda=no_conda)
    if flavor_backend is None:
        raise Exception("No suitable flavor backend was found for the model.")
    return flavor_backend

