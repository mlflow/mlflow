import logging
import os

import click
import yaml
from packaging.requirements import InvalidRequirement, Requirement

from mlflow.models import python_api
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import cli_args
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _is_pip_deps,
)

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
@cli_args.ENV_MANAGER
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
@click.argument("reqs_string", type=str)
def update_pip_requirements(model_uri, operation, reqs_string):
    """
    Updates the requirements.txt and conda.yaml files in the model's directory with a specified
    comma-separated string of requirements. The provided requirements will be added to the list
    of existing requirements. Any overlaps will be overwritten by the provided requirements.

    Example usages:

    mlflow models update-pip-requirements -m /path/to/model add "numpy==1.20.0,pandas==1.2.0"

    mlflow models update-pip-requirements -m /path/to/model remove "numpy,pandas"
    """
    if not os.path.isdir(model_uri):
        raise click.ClickException(f"Model directory not found at {model_uri}")

    conda_filepath = os.path.join(model_uri, _CONDA_ENV_FILE_NAME)
    if not os.path.exists(conda_filepath):
        raise click.ClickException(f"Conda environment file not found at {conda_filepath}")

    requirements_filepath = os.path.join(model_uri, _REQUIREMENTS_FILE_NAME)
    if not os.path.exists(requirements_filepath):
        raise click.ClickException(f"Requirements file not found at {requirements_filepath}")

    with open(conda_filepath) as file:
        conda_dict = yaml.safe_load(file)
    conda_pip_deps_idx = None
    conda_deps = conda_dict.get("dependencies", [])
    for idx, dep in enumerate(conda_deps):
        if _is_pip_deps(dep):
            # return an index so we can pop from the dep list.
            # this makes it easier to append it back later
            conda_pip_deps_idx = idx

    # treat conda YAML file as the source of truth for pip dependencies.
    # note: if conda.yaml and requirements.txt are out of sync, then
    # requirements.txt will be overwritten by the requirements from conda.yaml
    conda_pip_deps = [Requirement(s) for s in conda_deps.pop(conda_pip_deps_idx)["pip"]]
    conda_reqs_dict = {req.name: str(req) for req in conda_pip_deps}
    try:
        new_reqs = [Requirement(s.strip().lower()) for s in reqs_string.split(",")]
    except InvalidRequirement as e:
        raise click.ClickException(f"Received invalid requirement: {e}")

    _logger.info(f"Old requirements: {list(conda_reqs_dict.values())}")
    if operation == "remove":
        for req in new_reqs:
            if req.name not in conda_reqs_dict:
                _logger.info(f'Requirement "{req.name}" not found in conda.yaml, ignoring')
            conda_reqs_dict.pop(req.name, None)
    elif operation == "add":
        new_reqs_dict = {req.name: str(req) for req in new_reqs}
        conda_reqs_dict.update(new_reqs_dict)

    updated_reqs = list(conda_reqs_dict.values())
    _logger.info(f"New requirements: {updated_reqs}")
    conda_dict["dependencies"].append({"pip": updated_reqs})

    with open(conda_filepath, "w") as file:
        yaml.dump(conda_dict, file)

    with open(requirements_filepath, "w") as file:
        file.write("\n".join(updated_reqs))
