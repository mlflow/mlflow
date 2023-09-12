import contextlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import warnings
from datetime import timedelta

import click
from click import UsageError

import mlflow.db
import mlflow.deployments.cli
import mlflow.experiments
import mlflow.runs
import mlflow.store.artifact.cli
from mlflow import projects, version
from mlflow.entities import ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.exceptions import InvalidUrlException, MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.tracking import _get_store
from mlflow.utils import cli_args
from mlflow.utils.logging_utils import eprint
from mlflow.utils.os import is_windows
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


@click.group(cls=AliasedGroup)
@click.version_option(version=version.VERSION)
def cli():
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
        eprint("Specify only one of 'experiment-name' or 'experiment-id' options.")
        sys.exit(1)

    param_dict = _user_args_to_dict(param_list)
    args_dict = _user_args_to_dict(docker_args, argument_type="A")

    if backend_config is not None and os.path.splitext(backend_config)[-1] != ".json":
        try:
            backend_config = json.loads(backend_config)
        except ValueError as e:
            eprint(f"Invalid backend config JSON. Parse error: {e}")
            raise
    if backend == "kubernetes":
        if backend_config is None:
            eprint("Specify 'backend_config' when using kubernetes mode.")
            sys.exit(1)
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
            eprint(
                f"Invalid format for -{argument_type} parameter: '{arg}'. "
                f"Use -{argument_type} name=value."
            )
            sys.exit(1)
        if name in user_dict:
            eprint(f"Repeated parameter: '{name}'")
            sys.exit(1)
        user_dict[name] = value
    return user_dict


def _validate_server_args(gunicorn_opts=None, workers=None, waitress_opts=None):
    if sys.platform == "win32":
        if gunicorn_opts is not None or workers is not None:
            raise NotImplementedError(
                "waitress replaces gunicorn on Windows, "
                "cannot specify --gunicorn-opts or --workers"
            )
    else:
        if waitress_opts is not None:
            raise NotImplementedError(
                "gunicorn replaces waitress on non-Windows platforms, "
                "cannot specify --waitress-opts"
            )


def _validate_static_prefix(ctx, param, value):  # pylint: disable=unused-argument
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
    type=click.Choice([e.name for e in importlib.metadata.entry_points().get("mlflow.app", [])]),
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
        "Cannot be used with '--gunicorn-opts'. "
        "Unsupported on Windows."
    ),
)
def server(
    backend_store_uri,
    registry_store_uri,
    default_artifact_root,
    serve_artifacts,
    artifacts_only,
    artifacts_destination,
    host,
    port,
    workers,
    static_prefix,
    gunicorn_opts,
    waitress_opts,
    expose_prometheus,
    app_name,
    dev,
):
    """
    Run the MLflow tracking server.

    The server listens on http://localhost:5000 by default and only accepts connections
    from the local machine. To let the server accept connections from other machines, you will need
    to pass ``--host 0.0.0.0`` to listen on all network interfaces
    (or a specific interface address).
    """
    from mlflow.server import _run_server
    from mlflow.server.handlers import initialize_backend_stores

    if dev and is_windows():
        raise click.UsageError("'--dev' is not supported on Windows.")

    if dev and gunicorn_opts:
        raise click.UsageError("'--dev' and '--gunicorn-opts' cannot be specified together.")

    gunicorn_opts = "--log-level debug --reload" if dev else gunicorn_opts
    _validate_server_args(gunicorn_opts=gunicorn_opts, workers=workers, waitress_opts=waitress_opts)

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

    try:
        _run_server(
            backend_store_uri,
            registry_store_uri,
            default_artifact_root,
            serve_artifacts,
            artifacts_only,
            artifacts_destination,
            host,
            port,
            static_prefix,
            workers,
            gunicorn_opts,
            waitress_opts,
            expose_prometheus,
            app_name,
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
def gc(older_than, backend_store_uri, run_ids, experiment_ids):
    """
    Permanently delete runs in the `deleted` lifecycle stage from the specified backend store.
    This command deletes all artifacts and metadata associated with the specified runs.
    If the provided artifact URL is invalid, the artifact deletion will be bypassed,
    and the gc process will continue.
    """
    from mlflow.utils.time import get_current_time_millis

    backend_store = _get_store(backend_store_uri, None)
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

    deleted_run_ids_older_than = backend_store._get_deleted_runs(older_than=time_delta)
    if not run_ids:
        run_ids = deleted_run_ids_older_than
    else:
        run_ids = run_ids.split(",")

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
                "Run % is not in `deleted` lifecycle stage. Only runs in"
                " `deleted` lifecycle stage can be deleted." % run_id
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

# We are conditional loading these commands since the skinny client does
# not support them due to the pandas and numpy dependencies of MLflow Models
try:
    import mlflow.models.cli

    cli.add_command(mlflow.models.cli.commands)
except ImportError:
    pass

try:
    import mlflow.recipes.cli

    cli.add_command(mlflow.recipes.cli.commands)
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
