import logging
import click
import os

from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import cli_args
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import append_to_uri_path

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
@cli_args.ENABLE_MLSERVER
def serve(
    model_uri, port, host, workers, no_conda=False, install_mlflow=False, enable_mlserver=False
):
    """
    Serve a model saved with MLflow by launching a webserver on the specified host and port.
    The command supports models with the ``python_function`` or ``crate`` (R Function) flavor.
    For information about the input data formats accepted by the webserver, see the following
    documentation: https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    You can make requests to ``POST /invocations`` in pandas split- or record-oriented formats.

    Example:

    .. code-block:: bash

        $ mlflow models serve -m runs:/my-run-id/model-path &

        $ curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
            "columns": ["a", "b", "c"],
            "data": [[1, 2, 3], [4, 5, 6]]
        }'
    """
    return _get_flavor_backend(
        model_uri, no_conda=no_conda, workers=workers, install_mlflow=install_mlflow
    ).serve(model_uri=model_uri, port=port, host=host, enable_mlserver=enable_mlserver)


@commands.command("predict")
@cli_args.MODEL_URI
@click.option(
    "--input-path", "-i", default=None, help="CSV containing pandas DataFrame to predict against."
)
@click.option(
    "--output-path",
    "-o",
    default=None,
    help="File to output results to as json file. If not provided, output to stdout.",
)
@click.option(
    "--content-type",
    "-t",
    default="json",
    help="Content type of the input file. Can be one of {'json', 'csv'}.",
)
@click.option(
    "--json-format",
    "-j",
    default="split",
    help="Only applies if the content type is 'json'. Specify how the data is encoded.  "
    "Can be one of {'split', 'records'} mirroring the behavior of Pandas orient "
    "attribute. The default is 'split' which expects dict like data: "
    "{'index' -> [index], 'columns' -> [columns], 'data' -> [values]}, "
    "where index  is optional. For more information see "
    "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json"
    ".html",
)
@cli_args.NO_CONDA
@cli_args.INSTALL_MLFLOW
def predict(
    model_uri, input_path, output_path, content_type, json_format, no_conda, install_mlflow
):
    """
    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.
    """
    if content_type == "json" and json_format not in ("split", "records"):
        raise Exception("Unsupported json format '{}'.".format(json_format))
    return _get_flavor_backend(model_uri, no_conda=no_conda, install_mlflow=install_mlflow).predict(
        model_uri=model_uri,
        input_path=input_path,
        output_path=output_path,
        content_type=content_type,
        json_format=json_format,
    )


@commands.command("prepare-env")
@cli_args.MODEL_URI
@cli_args.NO_CONDA
@cli_args.INSTALL_MLFLOW
def prepare_env(model_uri, no_conda, install_mlflow):
    """
    Performs any preparation necessary to predict or serve the model, for example
    downloading dependencies or initializing a conda environment. After preparation,
    calling predict or serve should be fast.
    """
    return _get_flavor_backend(
        model_uri, no_conda=no_conda, install_mlflow=install_mlflow
    ).prepare_env(model_uri=model_uri)


@commands.command("build-docker")
@cli_args.MODEL_URI
@click.option("--name", "-n", default="mlflow-pyfunc-servable", help="Name to use for built image")
@cli_args.INSTALL_MLFLOW
@cli_args.ENABLE_MLSERVER
def build_docker(model_uri, name, install_mlflow, enable_mlserver):
    """
    Builds a Docker image whose default entrypoint serves the specified MLflow
    model at port 8080 within the container, using the 'python_function' flavor.

    For example, the following command builds a docker image named 'my-image-name' that serves the
    model from run 'some-run-uuid' at run-relative artifact path 'my-model':

    .. code:: bash

        mlflow models build-docker -m "runs:/some-run-uuid/my-model" -n "my-image-name"

    We can then serve the model, exposing it at port 5001 on the host via:

    .. code:: bash

        docker run -p 5001:8080 "my-image-name"

    NB: by default, the container will start nginx and gunicorn processes. If you don't need the
    nginx process to be started (for instance if you deploy your container to Google Cloud Run),
    you can disable it via the DISABLE_NGINX environment variable:

    .. code:: bash

        docker run -p 5001:8080 -e DISABLE_NGINX=true "my-image-name"


    See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html for more information on the
    'python_function' flavor.
    """
    mlflow_home = os.environ.get("MLFLOW_HOME", None)
    _get_flavor_backend(model_uri, docker_build=True).build_image(
        model_uri,
        name,
        mlflow_home=mlflow_home,
        install_mlflow=install_mlflow,
        enable_mlserver=enable_mlserver,
    )


def _get_flavor_backend(model_uri, **kwargs):
    from mlflow.models.flavor_backend_registry import get_flavor_backend

    with TempDir() as tmp:
        if ModelsArtifactRepository.is_models_uri(model_uri):
            underlying_model_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
        else:
            underlying_model_uri = model_uri
        local_path = _download_artifact_from_uri(
            append_to_uri_path(underlying_model_uri, MLMODEL_FILE_NAME), output_path=tmp.path()
        )
        model = Model.load(local_path)
    flavor_name, flavor_backend = get_flavor_backend(model, **kwargs)
    if flavor_backend is None:
        raise Exception("No suitable flavor backend was found for the model.")
    _logger.info("Selected backend for flavor '%s'", flavor_name)
    return flavor_backend
