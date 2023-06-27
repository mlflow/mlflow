"""
Definitions of click options shared by several CLI commands.
"""
import click
import warnings

from mlflow.utils import env_manager as _EnvManager
from mlflow.environment_variables import MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING

MODEL_PATH = click.option(
    "--model-path",
    "-m",
    default=None,
    metavar="PATH",
    required=True,
    help="Path to the model. The path is relative to the run with the given "
    "run-id or local filesystem path without run-id.",
)

_model_uri_help_string = (
    "URI to the model. A local path, a 'runs:/' URI, or a"
    " remote storage URI (e.g., an 's3://' URI). For more information"
    " about supported remote URIs for model artifacts, see"
    " https://mlflow.org/docs/latest/tracking.html#artifact-stores"
)

MODEL_URI_BUILD_DOCKER = click.option(
    "--model-uri",
    "-m",
    metavar="URI",
    default=None,
    required=False,
    help="[Optional] " + _model_uri_help_string,
)

MODEL_URI = click.option(
    "--model-uri",
    "-m",
    metavar="URI",
    required=True,
    help=_model_uri_help_string,
)

MLFLOW_HOME = click.option(
    "--mlflow-home",
    default=None,
    metavar="PATH",
    help="Path to local clone of MLflow project. Use for development only.",
)

RUN_ID = click.option(
    "--run-id",
    "-r",
    default=None,
    required=False,
    metavar="ID",
    help="ID of the MLflow run that generated the referenced content.",
)


def _resolve_env_manager(_, __, env_manager):
    if env_manager is not None:
        _EnvManager.validate(env_manager)
        if env_manager == _EnvManager.CONDA and not MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING.get():
            warnings.warn(
                (
                    "Use of conda is discouraged. If you use it, please ensure that your use of "
                    "conda complies with Anaconda's terms of service "
                    "(https://legal.anaconda.com/policies/en/?name=terms-of-service). "
                    "virtualenv is the recommended tool for environment reproducibility. "
                    f"To suppress this warning, set the {MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING} "
                    "environment variable to 'TRUE'."
                ),
                UserWarning,
                stacklevel=2,
            )
        return env_manager

    return None


def _create_env_manager_option(help_string):
    return click.option(
        "--env-manager",
        default=None,
        type=click.UNPROCESSED,
        callback=_resolve_env_manager,
        help=help_string,
    )


ENV_MANAGER = _create_env_manager_option(
    # '\b' prevents rewrapping text:
    # https://click.palletsprojects.com/en/8.1.x/documentation/#preventing-rewrapping
    help_string="""
If specified, create an environment for MLmodel using the specified
environment manager. The following values are supported:

\b
- local: use the local environment
- virtualenv: use virtualenv (and pyenv for Python version management)
- conda: use conda

If unspecified, default to virtualenv.
""",
)

ENV_MANAGER_PROJECTS = _create_env_manager_option(
    help_string="""
If specified, create an environment for MLproject using the specified
environment manager. The following values are supported:

\b
- local: use the local environment
- virtualenv: use virtualenv (and pyenv for Python version management)
- conda: use conda

If unspecified, the appropriate environment manager is automatically selected based on
the project configuration. For example, if `MLproject.yaml` contains a `python_env` key,
virtualenv is used.
""",
)


INSTALL_MLFLOW = click.option(
    "--install-mlflow",
    is_flag=True,
    default=False,
    help="If specified and there is a conda or virtualenv environment to be activated "
    "mlflow will be installed into the environment after it has been "
    "activated. The version of installed mlflow will be the same as "
    "the one used to invoke this command.",
)

HOST = click.option(
    "--host",
    "-h",
    envvar="MLFLOW_HOST",
    metavar="HOST",
    default="127.0.0.1",
    help="The network address to listen on (default: 127.0.0.1). "
    "Use 0.0.0.0 to bind to all addresses if you want to access the tracking "
    "server from other machines.",
)

PORT = click.option(
    "--port",
    "-p",
    envvar="MLFLOW_PORT",
    default=5000,
    help="The port to listen on (default: 5000).",
)

TIMEOUT = click.option(
    "--timeout",
    "-t",
    envvar="MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT",
    default=60,
    help="Timeout in seconds to serve a request (default: 60).",
)

# We use None to disambiguate manually selecting "4"
WORKERS = click.option(
    "--workers",
    "-w",
    envvar="MLFLOW_WORKERS",
    default=None,
    help="Number of gunicorn worker processes to handle requests (default: 1).",
)

ENABLE_MLSERVER = click.option(
    "--enable-mlserver",
    is_flag=True,
    default=False,
    help=(
        "Enable serving with MLServer through the v2 inference protocol. "
        "You can use environment variables to configure MLServer. "
        "(See https://mlserver.readthedocs.io/en/latest/reference/settings.html)"
    ),
)

ARTIFACTS_DESTINATION = click.option(
    "--artifacts-destination",
    envvar="MLFLOW_ARTIFACTS_DESTINATION",
    metavar="URI",
    default="./mlartifacts",
    help=(
        "The base artifact location from which to resolve artifact upload/download/list requests "
        "(e.g. 's3://my-bucket'). Defaults to a local './mlartifacts' directory. This option only "
        "applies when the tracking server is configured to stream artifacts and the experiment's "
        "artifact root location is http or mlflow-artifacts URI."
    ),
)

SERVE_ARTIFACTS = click.option(
    "--serve-artifacts/--no-serve-artifacts",
    envvar="MLFLOW_SERVE_ARTIFACTS",
    is_flag=True,
    default=True,
    help="Enables serving of artifact uploads, downloads, and list requests "
    "by routing these requests to the storage location that is specified by "
    "'--artifacts-destination' directly through a proxy. The default location that "
    "these requests are served from is a local './mlartifacts' directory which can be "
    "overridden via the '--artifacts-destination' argument. To disable artifact serving, "
    "specify `--no-serve-artifacts`. Default: True",
)

NO_CONDA = click.option(
    "--no-conda",
    is_flag=True,
    help="If specified, use local environment.",
)
