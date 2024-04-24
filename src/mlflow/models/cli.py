import logging

import click
from packaging.requirements import InvalidRequirement, Requirement

from mlflow.models import python_api
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import update_model_requirements
from mlflow.utils import cli_args
from mlflow.utils import env_manager as _EnvManager

_logger = logging.getLogger(__name__)


@click.group("models")
def commands():
    """
    Deploy MLflow models locally.

    To deploy a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """


@commands.command("serve")
@cli_args.MODEL_URI
@cli_args.PORT
@cli_args.HOST
@cli_args.TIMEOUT
@cli_args.WORKERS
@cli_args.ENV_MANAGER
@cli_args.NO_CONDA
@cli_args.INSTALL_MLFLOW
@cli_args.ENABLE_MLSERVER
def serve(
    model_uri,
    port,
    host,
    timeout,
    workers,
    env_manager=None,
    no_conda=False,
    install_mlflow=False,
    enable_mlserver=False,
):
    """
    Serve a model saved with MLflow by launching a webserver on the specified host and port.
    The command supports models with the ``python_function`` or ``crate`` (R Function) flavor.
    For information about the input data formats accepted by the webserver, see the following
    documentation: https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    .. warning::

        Models built using MLflow 1.x will require adjustments to the endpoint request payload
        if executed in an environment that has MLflow 2.x installed. In 1.x, a request payload
        was in the format: ``{'columns': [str], 'data': [[...]]}``. 2.x models require
        payloads that are defined by the structural-defining keys of either ``dataframe_split``,
        ``instances``, ``inputs`` or ``dataframe_records``. See the examples below for
        demonstrations of the changes to the invocation API endpoint in 2.0.

    .. note::

        Requests made in pandas DataFrame structures can be made in either `split` or `records`
        oriented formats.
        See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html for
        detailed information on orientation formats for converting a pandas DataFrame to json.

    Example:

    .. code-block:: bash

        $ mlflow models serve -m runs:/my-run-id/model-path &

        # records orientation input format for serializing a pandas DataFrame
        $ curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
            "dataframe_records": [{"a":1, "b":2}, {"a":3, "b":4}, {"a":5, "b":6}]
        }'

        # split orientation input format for serializing a pandas DataFrame
        $ curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
            "dataframe_split": {"columns": ["a", "b"],
                                "index": [0, 1, 2],
                                "data": [[1, 2], [3, 4], [5, 6]]}
        }'

        # inputs format for List submission of array, tensor, or DataFrame data
        $ curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
            "inputs": [[1, 2], [3, 4], [5, 6]]
        }'

        # instances format for submission of Tensor data
        curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
            "instances": [
                {"a": "t1", "b": [1, 2, 3]},
                {"a": "t2", "b": [4, 5, 6]},
                {"a": "t3", "b": [7, 8, 9]}
            ]
        }'

    """
    env_manager = _EnvManager.LOCAL if no_conda else env_manager

    return get_flavor_backend(
        model_uri, env_manager=env_manager, workers=workers, install_mlflow=install_mlflow
    ).serve(
        model_uri=model_uri, port=port, host=host, timeout=timeout, enable_mlserver=enable_mlserver
    )


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
@cli_args.ENV_MANAGER
@cli_args.INSTALL_MLFLOW
@click.option(
    "--pip-requirements-override",
    "-r",
    default=None,
    help="Specify packages and versions to override the dependencies defined "
    "in the model. Must be a comma-separated string like x==y,z==a.",
)
def predict(**kwargs):
    """
    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.
    """
    return python_api.predict(**kwargs)


@commands.command("prepare-env")
@cli_args.MODEL_URI
@cli_args.ENV_MANAGER
@cli_args.INSTALL_MLFLOW
def prepare_env(
    model_uri,
    env_manager,
    install_mlflow,
):
    """
    Performs any preparation necessary to predict or serve the model, for example
    downloading dependencies or initializing a conda environment. After preparation,
    calling predict or serve should be fast.
    """
    return get_flavor_backend(
        model_uri, env_manager=env_manager, install_mlflow=install_mlflow
    ).prepare_env(model_uri=model_uri)


@commands.command("generate-dockerfile")
@cli_args.MODEL_URI_BUILD_DOCKER
@click.option(
    "--output-directory",
    "-d",
    default="mlflow-dockerfile",
    help="Output directory where the generated Dockerfile is stored.",
)
@cli_args.ENV_MANAGER_DOCKERFILE
@cli_args.MLFLOW_HOME
@cli_args.INSTALL_JAVA
@cli_args.INSTALL_MLFLOW
@cli_args.ENABLE_MLSERVER
def generate_dockerfile(
    model_uri,
    output_directory,
    env_manager,
    mlflow_home,
    install_java,
    install_mlflow,
    enable_mlserver,
):
    """
    Generates a directory with Dockerfile whose default entrypoint serves an MLflow model at port
    8080 using the python_function flavor. The generated Dockerfile is written to the specified
    output directory, along with the model (if specified). This Dockerfile defines an image that
    is equivalent to the one produced by ``mlflow models build-docker``.
    """
    if model_uri:
        _logger.info("Generating Dockerfile for model %s", model_uri)
    else:
        _logger.info("Generating Dockerfile")
    backend = get_flavor_backend(model_uri, docker_build=True, env_manager=env_manager)
    if backend.can_build_image():
        backend.generate_dockerfile(
            model_uri,
            output_directory,
            mlflow_home=mlflow_home,
            install_java=install_java,
            install_mlflow=install_mlflow,
            enable_mlserver=enable_mlserver,
        )
        _logger.info("Generated Dockerfile in directory %s", output_directory)
    else:
        _logger.error(
            "Cannot build docker image for selected backend",
            extra={"backend": backend.__class__.__name__},
        )
        raise NotImplementedError("Cannot build docker image for selected backend")


@commands.command("build-docker")
@cli_args.MODEL_URI_BUILD_DOCKER
@click.option("--name", "-n", default="mlflow-pyfunc-servable", help="Name to use for built image")
@cli_args.ENV_MANAGER
@cli_args.MLFLOW_HOME
@cli_args.INSTALL_JAVA
@cli_args.INSTALL_MLFLOW
@cli_args.ENABLE_MLSERVER
def build_docker(**kwargs):
    """
    Builds a Docker image whose default entrypoint serves an MLflow model at port 8080, using the
    python_function flavor. The container serves the model referenced by ``--model-uri``, if
    specified when ``build-docker`` is called. If ``--model-uri`` is not specified when build_docker
    is called, an MLflow Model directory must be mounted as a volume into the /opt/ml/model
    directory in the container.

    Building a Docker image with ``--model-uri``:

    .. code:: bash

        # Build a Docker image named 'my-image-name' that serves the model from run 'some-run-uuid'
        # at run-relative artifact path 'my-model'
        mlflow models build-docker --model-uri "runs:/some-run-uuid/my-model" --name "my-image-name"
        # Serve the model
        docker run -p 5001:8080 "my-image-name"

    Building a Docker image without ``--model-uri``:

    .. code:: bash

        # Build a generic Docker image named 'my-image-name'
        mlflow models build-docker --name "my-image-name"
        # Mount the model stored in '/local/path/to/artifacts/model' and serve it
        docker run --rm -p 5001:8080 -v /local/path/to/artifacts/model:/opt/ml/model "my-image-name"

    .. important::

        Since MLflow 2.10.1, the Docker image built with ``--model-uri`` does **not install Java**
        for improved performance, unless the model flavor is one of ``["johnsnowlabs", "h2o",
        "mleap", "spark"]``. If you need to install Java for other flavors, e.g. custom Python model
        that uses SparkML, please specify the ``--install-java`` flag to enforce Java installation.

    .. warning::

        The image built without ``--model-uri`` doesn't support serving models with RFunc / Java
        MLeap model server.

    NB: by default, the container will start nginx and gunicorn processes. If you don't need the
    nginx process to be started (for instance if you deploy your container to Google Cloud Run),
    you can disable it via the DISABLE_NGINX environment variable:

    .. code:: bash

        docker run -p 5001:8080 -e DISABLE_NGINX=true "my-image-name"

    See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html for more information on the
    'python_function' flavor.
    """
    python_api.build_docker(**kwargs)


@commands.command("update-pip-requirements")
@cli_args.MODEL_URI
@click.argument("operation", type=click.Choice(["add", "remove"]))
@click.argument("requirement_strings", type=str, nargs=-1)
def update_pip_requirements(model_uri, operation, requirement_strings):
    """
    Add or remove requirements from a model's conda.yaml and requirements.txt files.
    If using a remote tracking server, please make sure to set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.

    REQUIREMENT_STRINGS is a list of pip requirements specifiers.
    See below for examples.

    Sample usage:

    .. code::

        # Add requirements using the model's "runs:/" URI

        mlflow models update-pip-requirements -m runs:/<run_id>/<model_path> \\
            add "pandas==1.0.0" "scikit-learn" "mlflow >= 2.8, != 2.9.0"

        # Remove requirements from a local model

        mlflow models update-pip-requirements -m /path/to/local/model \\
            remove "torchvision" "pydantic"

    Note that model registry URIs (i.e. URIs in the form ``models:/``) are not
    supported, as artifacts in the model registry are intended to be read-only.
    Editing requirements is read-only artifact repositories is also not supported.

    If adding requirements, the function will overwrite any existing requirements
    that overlap, or else append the new requirements to the existing list.

    If removing requirements, the function will ignore any version specifiers,
    and remove all the specified package names. Any requirements that are not
    found in the existing files will be ignored.
    """
    try:
        requirements = [Requirement(s.strip().lower()) for s in requirement_strings]
    except InvalidRequirement as e:
        raise click.BadArgumentUsage(f"Invalid requirement: {e}")

    update_model_requirements(model_uri, operation, requirements)

    _logger.info(f"Successfully updated the requirements for the model at {model_uri}!")
