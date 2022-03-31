"""
Definitions of click options shared by several CLI commands.
"""
import click
import warnings


from mlflow.utils.environment import EnvManager

MODEL_PATH = click.option(
    "--model-path",
    "-m",
    default=None,
    metavar="PATH",
    required=True,
    help="Path to the model. The path is relative to the run with the given "
    "run-id or local filesystem path without run-id.",
)

MODEL_URI = click.option(
    "--model-uri",
    "-m",
    default=None,
    metavar="URI",
    required=True,
    help="URI to the model. A local path, a 'runs:/' URI, or a"
    " remote storage URI (e.g., an 's3://' URI). For more information"
    " about supported remote URIs for model artifacts, see"
    " https://mlflow.org/docs/latest/tracking.html"
    "#artifact-stores",
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

NO_CONDA = click.option(
    "--no-conda",
    is_flag=True,
    help="[Deprecated] If specified, will assume that MLmodel/MLproject is running within "
    "a Conda environment with the necessary dependencies for "
    "the current project instead of attempting to create a new "
    "conda environment.",
)


def _env_manager_callback(_, __, value):
    """
    Validates the value of `--env-manager` and converts it to the corresponding member of
    `EnvManager`.
    """
    if value is None:
        return value

    allowed_values = [e.value for e in EnvManager]
    if value not in allowed_values:
        raise click.BadParameter(f"Expected one of {allowed_values} but got '{value}'")

    return EnvManager[value.upper()]


ENV_MANAGER = click.option(
    "--env-manager",
    default=None,
    required=False,
    type=click.UNPROCESSED,
    callback=_env_manager_callback,
    help="If specified, create an environment for MLmodel/MLproject using the specified "
    "environment manager. Valid values are ['local', 'conda']. If unspecified, default to "
    "'conda'.",
)


def _get_env_manager(no_conda, env_manager):
    # Both `--no-conda` and `--env-manager` are specified
    if no_conda and env_manager is not None:
        raise Exception(
            "`--no-conda` (deprecated) and `--env-manager` cannot be specified at the same time."
        )
    # Only `--no-conda` is specified
    elif no_conda:
        warnings.warn(
            (
                "`--no-conda` is deprecated and will be removed in a future MLflow release. "
                "Use `--env-manager=local` instead."
            ),
            FutureWarning,
            stacklevel=2,
        )
        return EnvManager.LOCAL
    # Neither `--no-conda` nor `--env-manager` is specified
    elif env_manager is None:
        return EnvManager.CONDA

    return env_manager


INSTALL_MLFLOW = click.option(
    "--install-mlflow",
    is_flag=True,
    default=False,
    help="If specified and there is a conda environment to be activated "
    "mlflow will be installed into the environment after it has been"
    " activated. The version of installed mlflow will be the same as"
    "the one used to invoke this command.",
)

HOST = click.option(
    "--host",
    "-h",
    metavar="HOST",
    default="127.0.0.1",
    help="The network address to listen on (default: 127.0.0.1). "
    "Use 0.0.0.0 to bind to all addresses if you want to access the tracking "
    "server from other machines.",
)

PORT = click.option("--port", "-p", default=5000, help="The port to listen on (default: 5000).")

# We use None to disambiguate manually selecting "4"
WORKERS = click.option(
    "--workers",
    "-w",
    default=None,
    help="Number of gunicorn worker processes to handle requests (default: 4).",
)

ENABLE_MLSERVER = click.option(
    "--enable-mlserver",
    is_flag=True,
    default=False,
    help="Enable serving with MLServer through the v2 inference protocol.",
)

ARTIFACTS_DESTINATION = click.option(
    "--artifacts-destination",
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
    "--serve-artifacts",
    is_flag=True,
    default=False,
    help="If specified, enables serving of artifact uploads, downloads, and list requests "
    "by routing these requests to the storage location that is specified by "
    "'--artifact-destination' directly through a proxy. The default location that "
    "these requests are served from is a local './mlartifacts' directory which can be "
    "overridden via the '--artifacts-destination' argument. "
    "Default: False",
)
