import contextlib
import json
import logging
import os
import re
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import click
from click import UsageError
from dotenv import load_dotenv

import mlflow.db
import mlflow.deployments.cli
import mlflow.experiments
import mlflow.runs
import mlflow.store.artifact.cli
from mlflow import ai_commands, projects, version
from mlflow.entities import ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.exceptions import InvalidUrlException, MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.tracking import _get_store
from mlflow.tracking._tracking_service.utils import is_tracking_uri_set, set_tracking_uri
from mlflow.utils import cli_args
from mlflow.utils.logging_utils import eprint
from mlflow.utils.os import is_windows
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.process import ShellCommandException
from mlflow.utils.server_cli_utils import (
    artifacts_only_config_validation,
    resolve_default_artifact_root,
)

_logger = logging.getLogger(__name__)


class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        # `mlflow ui` is an alias for `mlflow server`
        cmd_name = "server" if cmd_name == "ui" else cmd_name
        return super().get_command(ctx, cmd_name)


def _load_env_file(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """
    Click callback to load environment variables from a dotenv file.

    This function is designed to be used as an eager callback for the --env-file option,
    ensuring that environment variables are loaded before any command execution.
    """
    if value is not None:
        env_path = Path(value)
        if not env_path.exists():
            raise click.BadParameter(f"Environment file '{value}' does not exist.")

        # Load the environment file
        # override=False means existing environment variables take precedence
        load_dotenv(env_path, override=False)

        # Log that we've loaded the env file (using click.echo for CLI output)
        click.echo(f"Loaded environment variables from: {value}")

    return value


@click.group(cls=AliasedGroup)
@click.version_option(version=version.VERSION)
@click.option(
    "--env-file",
    type=click.Path(exists=False),
    callback=_load_env_file,
    expose_value=True,
    is_eager=True,
    help="Load environment variables from a dotenv file before executing the command. "
    "Variables in the file will be loaded but won't override existing environment variables.",
)
def cli(env_file):
    pass


@cli.command()
@click.argument("uri")
@click.option(
    "--entry-point",
    "-e",
    metavar="NAME",
    default="main",
    help="Entry point within project. [default: main]. If the entry point is not found, "
    "attempts to run the project file with the specified name as a script, "
    "using 'python' to run .py files and the default shell (specified by "
    "environment variable $SHELL) to run .sh files",
)
@click.option(
    "--version",
    "-v",
    metavar="VERSION",
    help="Version of the project to run, as a Git commit reference for Git projects.",
)
@click.option(
    "--param-list",
    "-P",
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the run, of the form -P name=value. Provided parameters that "
    "are not in the list of parameters for an entry point will be passed to the "
    "corresponding entry point as command-line arguments in the form `--name value`",
)
@click.option(
    "--docker-args",
    "-A",
    metavar="NAME=VALUE",
    multiple=True,
    help="A `docker run` argument or flag, of the form -A name=value (e.g. -A gpus=all) "
    "or -A name (e.g. -A t). The argument will then be passed as "
    "`docker run --name value` or `docker run --name` respectively. ",
)
@click.option(
    "--experiment-name",
    envvar=MLFLOW_EXPERIMENT_NAME.name,
    help="Name of the experiment under which to launch the run. If not "
    "specified, 'experiment-id' option will be used to launch run.",
)
@click.option(
    "--experiment-id",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    help="ID of the experiment under which to launch the run.",
)
# TODO: Add tracking server argument once we have it working.
@click.option(
    "--backend",
    "-b",
    metavar="BACKEND",
    default="local",
    help="Execution backend to use for run. Supported values: 'local', 'databricks', "
    "kubernetes (experimental). Defaults to 'local'. If running against "
    "Databricks, will run against a Databricks workspace determined as follows: "
    "if a Databricks tracking URI of the form 'databricks://profile' has been set "
    "(e.g. by setting the MLFLOW_TRACKING_URI environment variable), will run "
    "against the workspace specified by <profile>. Otherwise, runs against the "
    "workspace specified by the default Databricks CLI profile. See "
    "https://github.com/databricks/databricks-cli for more info on configuring a "
    "Databricks CLI profile.",
)
@click.option(
    "--backend-config",
    "-c",
    metavar="FILE",
    help="Path to JSON file (must end in '.json') or JSON string which will be passed "
    "as config to the backend. The exact content which should be "
    "provided is different for each execution backend and is documented "
    "at https://www.mlflow.org/docs/latest/projects.html.",
)
@cli_args.ENV_MANAGER_PROJECTS
@click.option(
    "--storage-dir",
    envvar="MLFLOW_TMP_DIR",
    help="Only valid when ``backend`` is local. "
    "MLflow downloads artifacts from distributed URIs passed to parameters of "
    "type 'path' to subdirectories of storage_dir.",
)
@click.option(
    "--run-id",
    metavar="RUN_ID",
    help="If specified, the given run ID will be used instead of creating a new run. "
    "Note: this argument is used internally by the MLflow project APIs "
    "and should not be specified.",
)
@click.option(
    "--run-name",
    metavar="RUN_NAME",
    help="The name to give the MLflow Run associated with the project execution. If not specified, "
    "the MLflow Run name is left unset.",
)
@click.option(
    "--build-image",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Only valid for Docker projects. If specified, build a new Docker image that's based on "
        "the image specified by the `image` field in the MLproject file, and contains files in the "
        "project directory."
    ),
)
def run(
    uri,
    entry_point,
    version,
    param_list,
    docker_args,
    experiment_name,
    experiment_id,
    backend,
    backend_config,
    env_manager,
    storage_dir,
    run_id,
    run_name,
    build_image,
):
    """
    Run an MLflow project from the given URI.

    For local runs, the run will block until it completes.
    Otherwise, the project will run asynchronously.

    If running locally (the default), the URI can be either a Git repository URI or a local path.
    If running on Databricks, the URI must be a Git repository.

    By default, Git projects run in a new working directory with the given parameters, while
    local projects run from the project's root directory.
    """
    if experiment_id is not None and experiment_name is not None:
        raise click.UsageError("Specify only one of 'experiment-name' or 'experiment-id' options.")

    param_dict = _user_args_to_dict(param_list)
    args_dict = _user_args_to_dict(docker_args, argument_type="A")

    if backend_config is not None and os.path.splitext(backend_config)[-1] != ".json":
        try:
            backend_config = json.loads(backend_config)
        except ValueError as e:
            raise click.UsageError(f"Invalid backend config JSON. Parse error: {e}") from e
    if backend == "kubernetes":
        if backend_config is None:
            raise click.UsageError("Specify 'backend_config' when using kubernetes mode.")
    try:
        projects.run(
            uri,
            entry_point,
            version,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            parameters=param_dict,
            docker_args=args_dict,
            backend=backend,
            backend_config=backend_config,
            env_manager=env_manager,
            storage_dir=storage_dir,
            synchronous=backend in ("local", "kubernetes") or backend is None,
            run_id=run_id,
            run_name=run_name,
            build_image=build_image,
        )
    except projects.ExecutionException as e:
        _logger.error("=== %s ===", e)
        sys.exit(1)


def _user_args_to_dict(arguments, argument_type="P"):
    user_dict = {}
    for arg in arguments:
        split = arg.split("=", maxsplit=1)
        # Docker arguments such as `t` don't require a value -> set to True if specified
        if len(split) == 1 and argument_type == "A":
            name = split[0]
            value = True
        elif len(split) == 2:
            name = split[0]
            value = split[1]
        else:
            raise click.UsageError(
                f"Invalid format for -{argument_type} parameter: '{arg}'. "
                f"Use -{argument_type} name=value."
            )
        if name in user_dict:
            raise click.UsageError(f"Repeated parameter: '{name}'")
        user_dict[name] = value
    return user_dict


def _validate_server_args(
    ctx=None,
    gunicorn_opts=None,
    workers=None,
    waitress_opts=None,
    uvicorn_opts=None,
    allowed_hosts=None,
    cors_allowed_origins=None,
    x_frame_options=None,
    disable_security_middleware=None,
):
    if sys.platform == "win32":
        if gunicorn_opts is not None:
            raise NotImplementedError(
                "gunicorn is not supported on Windows, cannot specify --gunicorn-opts"
            )

    num_server_opts_specified = sum(
        1 for opt in [gunicorn_opts, waitress_opts, uvicorn_opts] if opt is not None
    )
    if num_server_opts_specified > 1:
        raise click.UsageError(
            "Cannot specify multiple server options. Choose one of: "
            "'--gunicorn-opts', '--waitress-opts', or '--uvicorn-opts'."
        )

    using_flask_only = gunicorn_opts is not None or waitress_opts is not None
    # NB: Only check for security params that are explicitly passed via CLI (not env vars)
    # This allows Docker containers to set env vars while using gunicorn
    from click.core import ParameterSource

    security_params_specified = False
    if ctx:
        security_params_specified = any(
            [
                ctx.get_parameter_source("allowed_hosts") == ParameterSource.COMMANDLINE,
                ctx.get_parameter_source("cors_allowed_origins") == ParameterSource.COMMANDLINE,
                (
                    ctx.get_parameter_source("disable_security_middleware")
                    == ParameterSource.COMMANDLINE
                ),
            ]
        )

    if using_flask_only and security_params_specified:
        raise click.UsageError(
            "Security middleware parameters (--allowed-hosts, --cors-allowed-origins, "
            "--disable-security-middleware) are only supported with "
            "the default uvicorn server. They cannot be used with --gunicorn-opts or "
            "--waitress-opts. To use security features, run without specifying a server "
            "option (uses uvicorn by default) or explicitly use --uvicorn-opts."
        )


def _validate_static_prefix(ctx, param, value):
    """
    Validate that the static_prefix option starts with a "/" and does not end in a "/".
    Conforms to the callback interface of click documented at
    http://click.pocoo.org/5/options/#callbacks-for-validation.
    """
    if value is not None:
        if not value.startswith("/"):
            raise UsageError("--static-prefix must begin with a '/'.")
        if value.endswith("/"):
            raise UsageError("--static-prefix should not end with a '/'.")
    return value


@cli.command()
@click.pass_context
@click.option(
    "--backend-store-uri",
    envvar="MLFLOW_BACKEND_STORE_URI",
    metavar="PATH",
    default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    help="URI to which to persist experiment and run data. Acceptable URIs are "
    "SQLAlchemy-compatible database connection strings "
    "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
    "(e.g. 'file:///absolute/path/to/directory'). By default, data will be logged "
    "to the ./mlruns directory.",
)
@click.option(
    "--registry-store-uri",
    envvar="MLFLOW_REGISTRY_STORE_URI",
    metavar="URI",
    default=None,
    help="URI to which to persist registered models. Acceptable URIs are "
    "SQLAlchemy-compatible database connection strings (e.g. 'sqlite:///path/to/file.db'). "
    "If not specified, `backend-store-uri` is used.",
)
@click.option(
    "--default-artifact-root",
    envvar="MLFLOW_DEFAULT_ARTIFACT_ROOT",
    metavar="URI",
    default=None,
    help="Directory in which to store artifacts for any new experiments created. For tracking "
    "server backends that rely on SQL, this option is required in order to store artifacts. "
    "Note that this flag does not impact already-created experiments with any previous "
    "configuration of an MLflow server instance. "
    f"By default, data will be logged to the {DEFAULT_ARTIFACTS_URI} uri proxy if "
    "the --serve-artifacts option is enabled. Otherwise, the default location will "
    f"be {DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH}.",
)
@cli_args.SERVE_ARTIFACTS
@click.option(
    "--artifacts-only",
    envvar="MLFLOW_ARTIFACTS_ONLY",
    is_flag=True,
    default=False,
    help="If specified, configures the mlflow server to be used only for proxied artifact serving. "
    "With this mode enabled, functionality of the mlflow tracking service (e.g. run creation, "
    "metric logging, and parameter logging) is disabled. The server will only expose "
    "endpoints for uploading, downloading, and listing artifacts. "
    "Default: False",
)
@cli_args.ARTIFACTS_DESTINATION
@cli_args.HOST
@cli_args.PORT
@cli_args.WORKERS
@cli_args.ALLOWED_HOSTS
@cli_args.CORS_ALLOWED_ORIGINS
@cli_args.DISABLE_SECURITY_MIDDLEWARE
@cli_args.X_FRAME_OPTIONS
@click.option(
    "--static-prefix",
    envvar="MLFLOW_STATIC_PREFIX",
    default=None,
    callback=_validate_static_prefix,
    help="A prefix which will be prepended to the path of all static paths.",
)
@click.option(
    "--gunicorn-opts",
    envvar="MLFLOW_GUNICORN_OPTS",
    default=None,
    help="Additional command line options forwarded to gunicorn processes.",
)
@click.option(
    "--waitress-opts", default=None, help="Additional command line options for waitress-serve."
)
@click.option(
    "--uvicorn-opts",
    envvar="MLFLOW_UVICORN_OPTS",
    default=None,
    help="Additional command line options forwarded to uvicorn processes (used by default).",
)
@click.option(
    "--expose-prometheus",
    envvar="MLFLOW_EXPOSE_PROMETHEUS",
    default=None,
    help="Path to the directory where metrics will be stored. If the directory "
    "doesn't exist, it will be created. "
    "Activate prometheus exporter to expose metrics on /metrics endpoint.",
)
@click.option(
    "--app-name",
    default=None,
    type=click.Choice([e.name for e in get_entry_points("mlflow.app")]),
    show_default=True,
    help=(
        "Application name to be used for the tracking server. "
        "If not specified, 'mlflow.server:app' will be used."
    ),
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "If enabled, run the server with debug logging and auto-reload. "
        "Should only be used for development purposes. "
        "Cannot be used with '--gunicorn-opts' or '--uvicorn-opts'. "
        "Unsupported on Windows."
    ),
)
def server(
    ctx,
    backend_store_uri,
    registry_store_uri,
    default_artifact_root,
    serve_artifacts,
    artifacts_only,
    artifacts_destination,
    host,
    port,
    workers,
    allowed_hosts,
    cors_allowed_origins,
    disable_security_middleware,
    x_frame_options,
    static_prefix,
    gunicorn_opts,
    waitress_opts,
    expose_prometheus,
    app_name,
    dev,
    uvicorn_opts,
):
    """
    Run the MLflow tracking server with built-in security middleware.

    The server listens on http://localhost:5000 by default and only accepts connections
    from the local machine. To let the server accept connections from other machines, you will need
    to pass ``--host 0.0.0.0`` to listen on all network interfaces
    (or a specific interface address).

    See https://mlflow.org/docs/latest/tracking/server-security.html for detailed documentation
    and guidance on security configurations for the MLflow tracking server.
    """
    from mlflow.server import _run_server
    from mlflow.server.handlers import initialize_backend_stores

    # Get env_file from parent context
    env_file = ctx.parent.params.get("env_file") if ctx.parent else None

    if dev:
        if is_windows():
            raise click.UsageError("'--dev' is not supported on Windows.")
        if gunicorn_opts:
            raise click.UsageError("'--dev' and '--gunicorn-opts' cannot be specified together.")
        if uvicorn_opts:
            raise click.UsageError("'--dev' and '--uvicorn-opts' cannot be specified together.")
        if app_name:
            raise click.UsageError(
                "'--dev' cannot be used with '--app-name'. Development mode with auto-reload "
                "is only supported for the default MLflow tracking server."
            )

        uvicorn_opts = "--reload --log-level debug"

    _validate_server_args(
        ctx=ctx,
        gunicorn_opts=gunicorn_opts,
        workers=workers,
        waitress_opts=waitress_opts,
        uvicorn_opts=uvicorn_opts,
        allowed_hosts=allowed_hosts,
        cors_allowed_origins=cors_allowed_origins,
        x_frame_options=x_frame_options,
        disable_security_middleware=disable_security_middleware,
    )

    if disable_security_middleware:
        os.environ["MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE"] = "true"
    else:
        if allowed_hosts:
            os.environ["MLFLOW_SERVER_ALLOWED_HOSTS"] = allowed_hosts
            if allowed_hosts == "*":
                click.echo(
                    "WARNING: Accepting ALL hosts. "
                    "This may leave the server vulnerable to DNS rebinding attacks."
                )

        if cors_allowed_origins:
            os.environ["MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"] = cors_allowed_origins
            if cors_allowed_origins == "*":
                click.echo(
                    "WARNING: Allowing ALL origins for CORS. "
                    "This allows ANY website to access your MLflow data. "
                    "This configuration is only recommended for local development."
                )

        if x_frame_options:
            os.environ["MLFLOW_SERVER_X_FRAME_OPTIONS"] = x_frame_options

    # Ensure that both backend_store_uri and default_artifact_uri are set correctly.
    if not backend_store_uri:
        backend_store_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    # the default setting of registry_store_uri is same as backend_store_uri
    if not registry_store_uri:
        registry_store_uri = backend_store_uri

    default_artifact_root = resolve_default_artifact_root(
        serve_artifacts, default_artifact_root, backend_store_uri
    )
    artifacts_only_config_validation(artifacts_only, backend_store_uri)

    try:
        initialize_backend_stores(backend_store_uri, registry_store_uri, default_artifact_root)
    except Exception as e:
        _logger.error("Error initializing backend store")
        _logger.exception(e)
        sys.exit(1)

    if disable_security_middleware:
        click.echo(
            "[MLflow] WARNING: Security middleware is DISABLED. "
            "Your MLflow server is vulnerable to various attacks.",
            err=True,
        )
    elif not allowed_hosts and not cors_allowed_origins:
        click.echo(
            "[MLflow] Security middleware enabled with default settings (localhost-only). "
            "To allow connections from other hosts, use --host 0.0.0.0 and configure "
            "--allowed-hosts and --cors-allowed-origins.",
            err=True,
        )
    else:
        parts = ["[MLflow] Security middleware enabled"]
        if allowed_hosts:
            hosts_list = allowed_hosts.split(",")[:3]
            if len(allowed_hosts.split(",")) > 3:
                hosts_list.append(f"and {len(allowed_hosts.split(',')) - 3} more")
            parts.append(f"Allowed hosts: {', '.join(hosts_list)}")
        if cors_allowed_origins:
            origins_list = cors_allowed_origins.split(",")[:3]
            if len(cors_allowed_origins.split(",")) > 3:
                origins_list.append(f"and {len(cors_allowed_origins.split(',')) - 3} more")
            parts.append(f"CORS origins: {', '.join(origins_list)}")
        click.echo(". ".join(parts) + ".", err=True)

    try:
        _run_server(
            file_store_path=backend_store_uri,
            registry_store_uri=registry_store_uri,
            default_artifact_root=default_artifact_root,
            serve_artifacts=serve_artifacts,
            artifacts_only=artifacts_only,
            artifacts_destination=artifacts_destination,
            host=host,
            port=port,
            static_prefix=static_prefix,
            workers=workers,
            gunicorn_opts=gunicorn_opts,
            waitress_opts=waitress_opts,
            expose_prometheus=expose_prometheus,
            app_name=app_name,
            uvicorn_opts=uvicorn_opts,
            env_file=env_file,
        )
    except ShellCommandException:
        eprint("Running the mlflow server failed. Please see the logs above for details.")
        sys.exit(1)


@cli.command(short_help="Permanently delete runs in the `deleted` lifecycle stage.")
@click.option(
    "--older-than",
    default=None,
    help="Optional. Remove run(s) older than the specified time limit. "
    "Specify a string in #d#h#m#s format. Float values are also supported. "
    "For example: --older-than 1d2h3m4s, --older-than 1.2d3h4m5s",
)
@click.option(
    "--backend-store-uri",
    metavar="PATH",
    default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    help="URI of the backend store from which to delete runs. Acceptable URIs are "
    "SQLAlchemy-compatible database connection strings "
    "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
    "(e.g. 'file:///absolute/path/to/directory'). By default, data will be deleted "
    "from the ./mlruns directory.",
)
@click.option(
    "--artifacts-destination",
    envvar="MLFLOW_ARTIFACTS_DESTINATION",
    metavar="URI",
    default=None,
    help=(
        "The base artifact location from which to resolve artifact upload/download/list requests "
        "(e.g. 's3://my-bucket'). This option only applies when the tracking server is configured "
        "to stream artifacts and the experiment's artifact root location is http or "
        "mlflow-artifacts URI. Otherwise, the default artifact location will be used."
    ),
)
@click.option(
    "--run-ids",
    default=None,
    help="Optional comma separated list of runs to be permanently deleted. If run ids"
    " are not specified, data is removed for all runs in the `deleted`"
    " lifecycle stage.",
)
@click.option(
    "--experiment-ids",
    default=None,
    help="Optional comma separated list of experiments to be permanently deleted including "
    "all of their associated runs. If experiment ids are not specified, data is removed for all "
    "experiments in the `deleted` lifecycle stage.",
)
@click.option(
    "--tracking-uri",
    default=os.environ.get("MLFLOW_TRACKING_URI"),
    help="Tracking URI to use for deleting 'deleted' runs e.g. http://127.0.0.1:8080",
)
def gc(older_than, backend_store_uri, artifacts_destination, run_ids, experiment_ids, tracking_uri):
    """
    Permanently delete runs in the `deleted` lifecycle stage from the specified backend store.
    This command deletes all artifacts and metadata associated with the specified runs.
    If the provided artifact URL is invalid, the artifact deletion will be bypassed,
    and the gc process will continue.

    .. attention::

        If you are running an MLflow tracking server with artifact proxying enabled,
        you **must** set the ``MLFLOW_TRACKING_URI`` environment variable before running
        this command. Otherwise, the ``gc`` command will not be able to resolve
        artifact URIs and will not be able to delete the associated artifacts.

    """
    from mlflow.utils.time import get_current_time_millis

    backend_store = _get_store(backend_store_uri, artifacts_destination)
    skip_experiments = False
    if not hasattr(backend_store, "_hard_delete_run"):
        raise MlflowException(
            "This cli can only be used with a backend that allows hard-deleting runs"
        )

    if not hasattr(backend_store, "_hard_delete_experiment"):
        warnings.warn(
            "The specified backend does not allow hard-deleting experiments. Experiments"
            " will be skipped.",
            FutureWarning,
            stacklevel=2,
        )
        skip_experiments = True

    time_delta = 0

    if older_than is not None:
        regex = re.compile(
            r"^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)"
            r"?((?P<seconds>[\.\d]+?)s)?$"
        )
        parts = regex.match(older_than)
        if parts is None:
            raise MlflowException(
                f"Could not parse any time information from '{older_than}'. "
                "Examples of valid strings: '8h', '2d8h5m20s', '2m4s'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
        time_delta = int(timedelta(**time_params).total_seconds() * 1000)

    if tracking_uri:
        set_tracking_uri(tracking_uri)

    if not is_tracking_uri_set():
        raise MlflowException(
            "Tracking URL is not set. Please set MLFLOW_TRACKING_URI environment variable "
            "or provide --tracking-uri cli option."
        )

    deleted_run_ids_older_than = backend_store._get_deleted_runs(older_than=time_delta)
    run_ids = run_ids.split(",") if run_ids else deleted_run_ids_older_than

    time_threshold = get_current_time_millis() - time_delta
    if not skip_experiments:
        if experiment_ids:
            experiment_ids = experiment_ids.split(",")
            experiments = [backend_store.get_experiment(id) for id in experiment_ids]

            # Ensure that the specified experiments are soft-deleted
            active_experiment_ids = [
                e.experiment_id for e in experiments if e.lifecycle_stage != LifecycleStage.DELETED
            ]
            if active_experiment_ids:
                raise MlflowException(
                    f"Experiments {active_experiment_ids} are not in the deleted lifecycle stage. "
                    "Only experiments in the deleted lifecycle stage can be hard-deleted.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            # Ensure that the specified experiments are old enough
            if older_than:
                non_old_experiment_ids = [
                    e.experiment_id
                    for e in experiments
                    if e.last_update_time is None or e.last_update_time >= time_threshold
                ]
                if non_old_experiment_ids:
                    raise MlflowException(
                        f"Experiments {non_old_experiment_ids} are not older than the required"
                        f"age. Only experiments older than {older_than} can be deleted.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
        else:
            filter_string = f"last_update_time < {time_threshold}" if older_than else None

            def fetch_experiments(token=None):
                page = backend_store.search_experiments(
                    view_type=ViewType.DELETED_ONLY,
                    filter_string=filter_string,
                    page_token=token,
                )
                return (page + fetch_experiments(page.token)) if page.token else page

            experiment_ids = [exp.experiment_id for exp in fetch_experiments()]

        def fetch_runs(token=None):
            page = backend_store.search_runs(
                experiment_ids=experiment_ids,
                filter_string="",
                run_view_type=ViewType.DELETED_ONLY,
                page_token=token,
            )
            return (page + fetch_runs(page.token)) if page.token else page

        run_ids.extend([run.info.run_id for run in fetch_runs()])

    for run_id in set(run_ids):
        run = backend_store.get_run(run_id)
        if run.info.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException(
                f"Run {run_id} is not in `deleted` lifecycle stage. Only runs in"
                " `deleted` lifecycle stage can be deleted."
            )
        # raise MlflowException if run_id is newer than older_than parameter
        if older_than and run_id not in deleted_run_ids_older_than:
            raise MlflowException(
                f"Run {run_id} is not older than the required age. "
                f"Only runs older than {older_than} can be deleted.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        # raise MlflowException if run_id is newer than older_than parameter
        if older_than and run_id not in deleted_run_ids_older_than:
            raise MlflowException(
                f"Run {run_id} is not older than the required age. "
                f"Only runs older than {older_than} can be deleted.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        artifact_repo = get_artifact_repository(run.info.artifact_uri)
        try:
            artifact_repo.delete_artifacts()
        except InvalidUrlException as iue:
            click.echo(
                click.style(
                    f"An exception {iue!r} was raised during the deletion of a model artifact",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"Unable to resolve the provided artifact URL: '{artifact_repo}'. "
                    "The gc process will continue and bypass artifact deletion. "
                    "Please ensure that the artifact exists "
                    "and consider manually deleting any unused artifacts. ",
                    fg="yellow",
                ),
            )
        backend_store._hard_delete_run(run_id)
        click.echo(f"Run with ID {run_id} has been permanently deleted.")

    if not skip_experiments:
        for experiment_id in experiment_ids:
            backend_store._hard_delete_experiment(experiment_id)
            click.echo(f"Experiment with ID {experiment_id} has been permanently deleted.")


@cli.command(short_help="Prints out useful information for debugging issues with MLflow.")
@click.option(
    "--mask-envs",
    is_flag=True,
    help=(
        "If set (the default behavior without setting this flag is not to obfuscate information), "
        'mask the MLflow environment variable values (e.g. `"MLFLOW_ENV_VAR": "***"`) '
        "in the output to prevent leaking sensitive information."
    ),
)
def doctor(mask_envs):
    mlflow.doctor(mask_envs)


cli.add_command(mlflow.deployments.cli.commands)
cli.add_command(mlflow.experiments.commands)
cli.add_command(mlflow.store.artifact.cli.commands)
cli.add_command(mlflow.runs.commands)
cli.add_command(mlflow.db.commands)

# Add traces CLI commands
from mlflow.cli import traces

cli.add_command(traces.commands)

# Add scorers CLI commands
from mlflow.cli import scorers

cli.add_command(scorers.commands)

# Add AI commands CLI
cli.add_command(ai_commands.commands)

try:
    from mlflow.mcp.cli import cli as mcp_cli

    cli.add_command(mcp_cli)
except ImportError:
    pass

# Add Claude Code integration commands
try:
    import mlflow.claude_code.cli

    cli.add_command(mlflow.claude_code.cli.commands)
except ImportError:
    pass

# We are conditional loading these commands since the skinny client does
# not support them due to the pandas and numpy dependencies of MLflow Models
try:
    import mlflow.models.cli

    cli.add_command(mlflow.models.cli.commands)
except ImportError:
    pass

try:
    import mlflow.sagemaker.cli

    cli.add_command(mlflow.sagemaker.cli.commands)
except ImportError:
    pass


with contextlib.suppress(ImportError):
    import mlflow.gateway.cli

    cli.add_command(mlflow.gateway.cli.commands)


if __name__ == "__main__":
    cli()
