import json
import os
import sys
import logging

import click
from click import UsageError

import mlflow.db
import mlflow.experiments
import mlflow.deployments.cli
import mlflow.projects as projects
import mlflow.runs
import mlflow.store.artifact.cli
from mlflow import tracking
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import _get_store
from mlflow.utils import cli_args
from mlflow.utils.annotations import experimental
from mlflow.utils.logging_utils import eprint
from mlflow.utils.process import ShellCommandException
from mlflow.utils.uri import is_local_uri
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
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
    envvar=tracking._EXPERIMENT_NAME_ENV_VAR,
    help="Name of the experiment under which to launch the run. If not "
    "specified, 'experiment-id' option will be used to launch run.",
)
@click.option(
    "--experiment-id",
    envvar=tracking._EXPERIMENT_ID_ENV_VAR,
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
@cli_args.NO_CONDA
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
    no_conda,
    storage_dir,
    run_id,
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
            eprint("Invalid backend config JSON. Parse error: %s" % e)
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
            use_conda=(not no_conda),
            storage_dir=storage_dir,
            synchronous=backend in ("local", "kubernetes") or backend is None,
            run_id=run_id,
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
                "Invalid format for -%s parameter: '%s'. "
                "Use -%s name=value." % (argument_type, arg, argument_type)
            )
            sys.exit(1)
        if name in user_dict:
            eprint("Repeated parameter: '%s'" % name)
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


@cli.command()
@click.option(
    "--backend-store-uri",
    metavar="PATH",
    default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    help="URI to which to persist experiment and run data. Acceptable URIs are "
    "SQLAlchemy-compatible database connection strings "
    "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
    "(e.g. 'file:///absolute/path/to/directory'). By default, data will be logged "
    "to the ./mlruns directory.",
)
@click.option(
    "--default-artifact-root",
    metavar="URI",
    default=None,
    help="Path to local directory to store artifacts, for new experiments. "
    "Note that this flag does not impact already-created experiments. "
    "Default: " + DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
)
@cli_args.PORT
@cli_args.HOST
def ui(backend_store_uri, default_artifact_root, port, host):
    """
    Launch the MLflow tracking UI for local viewing of run results. To launch a production
    server, use the "mlflow server" command instead.

    The UI will be visible at http://localhost:5000 by default, and only accept connections
    from the local machine. To let the UI server accept connections from other machines, you will
    need to pass ``--host 0.0.0.0`` to listen on all network interfaces (or a specific interface
    address).
    """
    from mlflow.server import _run_server
    from mlflow.server.handlers import initialize_backend_stores

    # Ensure that both backend_store_uri and default_artifact_uri are set correctly.
    if not backend_store_uri:
        backend_store_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    if not default_artifact_root:
        if is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        else:
            default_artifact_root = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    try:
        initialize_backend_stores(backend_store_uri, default_artifact_root)
    except Exception as e:
        _logger.error("Error initializing backend store")
        _logger.exception(e)
        sys.exit(1)

    # TODO: We eventually want to disable the write path in this version of the server.
    try:
        _run_server(backend_store_uri, default_artifact_root, host, port, None, 1)
    except ShellCommandException:
        eprint("Running the mlflow server failed. Please see the logs above for details.")
        sys.exit(1)


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
    metavar="PATH",
    default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    help="URI to which to persist experiment and run data. Acceptable URIs are "
    "SQLAlchemy-compatible database connection strings "
    "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
    "(e.g. 'file:///absolute/path/to/directory'). By default, data will be logged "
    "to the ./mlruns directory.",
)
@click.option(
    "--default-artifact-root",
    metavar="URI",
    default=None,
    help="Local or S3 URI to store artifacts, for new experiments. "
    "Note that this flag does not impact already-created experiments. "
    "Default: Within file store, if a file:/ URI is provided. If a sql backend is"
    " used, then this option is required.",
)
@cli_args.HOST
@cli_args.PORT
@cli_args.WORKERS
@click.option(
    "--static-prefix",
    default=None,
    callback=_validate_static_prefix,
    help="A prefix which will be prepended to the path of all static paths.",
)
@click.option(
    "--gunicorn-opts",
    default=None,
    help="Additional command line options forwarded to gunicorn processes.",
)
@click.option(
    "--waitress-opts", default=None, help="Additional command line options for waitress-serve."
)
@click.option(
    "--expose-prometheus",
    default=None,
    help="Path to the directory where metrics will be stored. If the directory "
    "doesn't exist, it will be created. "
    "Activate prometheus exporter to expose metrics on /metrics endpoint.",
)
def server(
    backend_store_uri,
    default_artifact_root,
    host,
    port,
    workers,
    static_prefix,
    gunicorn_opts,
    waitress_opts,
    expose_prometheus,
):
    """
    Run the MLflow tracking server.

    The server which listen on http://localhost:5000 by default, and only accept connections
    from the local machine. To let the server accept connections from other machines, you will need
    to pass ``--host 0.0.0.0`` to listen on all network interfaces
    (or a specific interface address).
    """
    from mlflow.server import _run_server
    from mlflow.server.handlers import initialize_backend_stores

    _validate_server_args(gunicorn_opts=gunicorn_opts, workers=workers, waitress_opts=waitress_opts)

    # Ensure that both backend_store_uri and default_artifact_uri are set correctly.
    if not backend_store_uri:
        backend_store_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    if not default_artifact_root:
        if is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        else:
            eprint(
                "Option 'default-artifact-root' is required, when backend store is not "
                "local file based."
            )
            sys.exit(1)

    try:
        initialize_backend_stores(backend_store_uri, default_artifact_root)
    except Exception as e:
        _logger.error("Error initializing backend store")
        _logger.exception(e)
        sys.exit(1)

    try:
        _run_server(
            backend_store_uri,
            default_artifact_root,
            host,
            port,
            static_prefix,
            workers,
            gunicorn_opts,
            waitress_opts,
            expose_prometheus,
        )
    except ShellCommandException:
        eprint("Running the mlflow server failed. Please see the logs above for details.")
        sys.exit(1)


@cli.command(short_help="Permanently delete runs in the `deleted` lifecycle stage.")
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
@experimental
def gc(backend_store_uri, run_ids):
    """
    Permanently delete runs in the `deleted` lifecycle stage from the specified backend store.
    This command deletes all artifacts and metadata associated with the specified runs.
    """
    backend_store = _get_store(backend_store_uri, None)
    if not hasattr(backend_store, "_hard_delete_run"):
        raise MlflowException(
            "This cli can only be used with a backend that allows hard-deleting runs"
        )
    if not run_ids:
        run_ids = backend_store._get_deleted_runs()
    else:
        run_ids = run_ids.split(",")

    for run_id in run_ids:
        run = backend_store.get_run(run_id)
        if run.info.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException(
                "Run % is not in `deleted` lifecycle stage. Only runs in "
                "`deleted` lifecycle stage can be deleted." % run_id
            )
        artifact_repo = get_artifact_repository(run.info.artifact_uri)
        artifact_repo.delete_artifacts()
        backend_store._hard_delete_run(run_id)
        print("Run with ID %s has been permanently deleted." % str(run_id))


cli.add_command(mlflow.deployments.cli.commands)
cli.add_command(mlflow.experiments.commands)
cli.add_command(mlflow.store.artifact.cli.commands)
cli.add_command(mlflow.runs.commands)
cli.add_command(mlflow.db.commands)

try:
    # pylint: disable=unused-import
    import mlflow.models.cli
    import mlflow.azureml.cli
    import mlflow.sagemaker.cli

    cli.add_command(mlflow.azureml.cli.commands)
    cli.add_command(mlflow.sagemaker.cli.commands)
    cli.add_command(mlflow.models.cli.commands)
except ImportError as e:
    # We are conditional loading these commands since the skinny client does
    # not support them due to the pandas and numpy dependencies of MLflow Models
    pass

if __name__ == "__main__":
    cli()
