"""
Definitions of click options shared by several CLI commands.
"""

import warnings

import click

from mlflow.environment_variables import MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING
from mlflow.utils import env_manager as _EnvManager

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


def _create_env_manager_option(help_string, default=None):
    return click.option(
        "--env-manager",
        default=default,
        type=click.UNPROCESSED,
        callback=_resolve_env_manager,
        help=help_string,
    )


ENV_MANAGER = _create_env_manager_option(
    default=_EnvManager.VIRTUALENV,
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
- uv: use uv
- conda: use conda

If unspecified, the appropriate environment manager is automatically selected based on
the project configuration. For example, if `MLproject.yaml` contains a `python_env` key,
virtualenv is used.
""",
)

ENV_MANAGER_DOCKERFILE = _create_env_manager_option(
    default=None,
    # '\b' prevents rewrapping text:
    # https://click.palletsprojects.com/en/8.1.x/documentation/#preventing-rewrapping
    help_string="""
If specified, create an environment for MLmodel using the specified
environment manager. The following values are supported:

\b
- local: use the local environment
- virtualenv: use virtualenv (and pyenv for Python version management)
- conda: use conda

If unspecified, default to None, then MLflow will automatically pick the env manager
based on the model's flavor configuration.
If model-uri is specified: if python version is specified in the flavor configuration
and no java installation is required, then we use local environment. Otherwise we use virtualenv.
If no model-uri is provided, we use virtualenv.
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
    help="The network interface to bind the server to (default: 127.0.0.1). "
    "This controls which network interfaces accept connections. "
    "Use '127.0.0.1' for local-only access, or '0.0.0.0' to allow connections from any network. "
    "NOTE: This is NOT a security setting - it only controls network binding. "
    "To restrict which clients can connect, use --allowed-hosts.",
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
    help="Number of worker processes to handle requests (default: 4).",
)

MODELS_WORKERS = click.option(
    "--workers",
    "-w",
    envvar="MLFLOW_MODELS_WORKERS",
    default=None,
    help="Number of uvicorn workers to handle requests when serving mlflow models (default: 1).",
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

INSTALL_JAVA = click.option(
    "--install-java",
    is_flag=False,
    flag_value=True,
    default=None,
    type=bool,
    help="Installs Java in the image if needed. Default is None, "
    "allowing MLflow to determine installation. Flavors requiring "
    "Java, such as Spark, enable this automatically. "
    "Note: This option only works with the UBUNTU base image; "
    "Python base images do not support Java installation.",
)

# Security-related options for MLflow server
ALLOWED_HOSTS = click.option(
    "--allowed-hosts",
    envvar="MLFLOW_SERVER_ALLOWED_HOSTS",
    default=None,
    help="Comma-separated list of allowed Host headers to prevent DNS rebinding attacks "
    "(default: localhost + private IPs). "
    "DNS rebinding allows attackers to trick your browser into accessing internal services. "
    "Examples: 'mlflow.company.com,10.0.0.100:5000'. "
    "Supports wildcards: 'mlflow.company.com,192.168.*,app-*.internal.com'. "
    "Use '*' to allow ALL hosts (not recommended for production). "
    "Default allows: localhost (all ports), private IPs (10.*, 192.168.*, 172.16-31.*). "
    "Set this when exposing MLflow beyond localhost to prevent host header attacks.",
)

CORS_ALLOWED_ORIGINS = click.option(
    "--cors-allowed-origins",
    envvar="MLFLOW_SERVER_CORS_ALLOWED_ORIGINS",
    default=None,
    help="Comma-separated list of allowed CORS origins to prevent cross-site request attacks "
    "(default: localhost origins on any port). "
    "CORS attacks allow malicious websites to make requests to your MLflow server using your "
    "credentials. Examples: 'https://app.company.com,https://notebook.company.com'. "
    "Default allows: http://localhost:* (any port), http://127.0.0.1:*, http://[::1]:*. "
    "Set this when you have web applications on different domains that need to access MLflow. "
    "Use '*' to allow ALL origins (DANGEROUS - only for development!).",
)

DISABLE_SECURITY_MIDDLEWARE = click.option(
    "--disable-security-middleware",
    envvar="MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE",
    is_flag=True,
    default=False,
    help="DANGEROUS: Disable all security middleware including CORS protection and host "
    "validation. This completely removes security protections and should only be used for "
    "testing. When disabled, your MLflow server is vulnerable to CORS attacks, DNS rebinding, "
    "and clickjacking. Instead, prefer configuring specific security settings with "
    "--cors-allowed-origins and --allowed-hosts.",
)

X_FRAME_OPTIONS = click.option(
    "--x-frame-options",
    envvar="MLFLOW_SERVER_X_FRAME_OPTIONS",
    default="SAMEORIGIN",
    help="X-Frame-Options header value for clickjacking protection. "
    "Options: 'SAMEORIGIN' (default - allows embedding only from same origin), "
    "'DENY' (prevents all embedding), 'NONE' (disables header - allows embedding from anywhere). "
    "Set to 'NONE' if you need to embed MLflow UI in iframes from different origins.",
)
