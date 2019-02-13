from __future__ import print_function

import json
import os
import sys
import logging

import click
from click import UsageError

import mlflow.azureml.cli
import mlflow.projects as projects
import mlflow.data
import mlflow.experiments
import mlflow.pyfunc.cli
import mlflow.rfunc.cli
import mlflow.sagemaker.cli
import mlflow.runs

from mlflow.entities.experiment import Experiment
from mlflow.tracking.utils import _is_local_uri
from mlflow.utils.logging_utils import eprint
from mlflow.utils.process import ShellCommandException
from mlflow.utils import cli_args
from mlflow.server import _run_server
from mlflow.store import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow import tracking
import mlflow.store.cli


_logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.argument("uri")
@click.option("--entry-point", "-e", metavar="NAME", default="main",
              help="Entry point within project. [default: main]. If the entry point is not found, "
                   "attempts to run the project file with the specified name as a script, "
                   "using 'python' to run .py files and the default shell (specified by "
                   "environment variable $SHELL) to run .sh files")
@click.option("--version", "-v", metavar="VERSION",
              help="Version of the project to run, as a Git commit reference for Git projects.")
@click.option("--param-list", "-P", metavar="NAME=VALUE", multiple=True,
              help="A parameter for the run, of the form -P name=value. Provided parameters that "
                   "are not in the list of parameters for an entry point will be passed to the "
                   "corresponding entry point as command-line arguments in the form `--name value`")
@click.option("--experiment-name", envvar=tracking._EXPERIMENT_NAME_ENV_VAR,
              help="Name of the experiment under which to launch the run. If not "
                   "specified, 'experiment-id' option will be used to launch run.")
@click.option("--experiment-id", envvar=tracking._EXPERIMENT_ID_ENV_VAR, type=click.INT,
              help="ID of the experiment under which to launch the run. Defaults to %s" %
                   Experiment.DEFAULT_EXPERIMENT_ID)
# TODO: Add tracking server argument once we have it working.
@click.option("--mode", "-m", metavar="MODE",
              help="Execution mode to use for run. Supported values: 'local' (runs project "
                   "locally) and 'databricks' (runs project on a Databricks cluster). "
                   "Defaults to 'local'. If running against Databricks, will run against a "
                   "Databricks workspace determined as follows: if a Databricks tracking URI "
                   "of the form 'databricks://profile' has been set (e.g. by setting "
                   "the MLFLOW_TRACKING_URI environment variable), will run against the "
                   "workspace specified by <profile>. Otherwise, runs against the workspace "
                   "specified by the default Databricks CLI profile. See "
                   "https://github.com/databricks/databricks-cli for more info on configuring a "
                   "Databricks CLI profile.")
@click.option("--cluster-spec", "-c", metavar="FILE",
              help="Path to JSON file (must end in '.json') or JSON string describing the cluster "
                   "to use when launching a run on Databricks. See "
                   "https://docs.databricks.com/api/latest/jobs.html#jobsclusterspecnewcluster for "
                   "more info. Note that MLflow runs are currently launched against a new cluster.")
@click.option("--git-username", metavar="USERNAME", envvar="MLFLOW_GIT_USERNAME",
              help="Username for HTTP(S) Git authentication.")
@click.option("--git-password", metavar="PASSWORD", envvar="MLFLOW_GIT_PASSWORD",
              help="Password for HTTP(S) Git authentication.")
@cli_args.NO_CONDA
@click.option("--storage-dir", envvar="MLFLOW_TMP_DIR",
              help="Only valid when `mode` is local."
                   "MLflow downloads artifacts from distributed URIs passed to parameters of "
                   "type 'path' to subdirectories of storage_dir.")
@click.option("--run-id", metavar="RUN_ID",
              help="If specified, the given run ID will be used instead of creating a new run. "
                   "Note: this argument is used internally by the MLflow project APIs "
                   "and should not be specified.")
def run(uri, entry_point, version, param_list, experiment_name, experiment_id, mode, cluster_spec,
        git_username, git_password, no_conda, storage_dir, run_id):
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

    param_dict = {}
    for s in param_list:
        index = s.find("=")
        if index == -1:
            eprint("Invalid format for -P parameter: '%s'. Use -P name=value." % s)
            sys.exit(1)
        name = s[:index]
        value = s[index + 1:]
        if name in param_dict:
            eprint("Repeated parameter: '%s'" % name)
            sys.exit(1)
        param_dict[name] = value
    cluster_spec_arg = cluster_spec
    if cluster_spec is not None and os.path.splitext(cluster_spec)[-1] != ".json":
        try:
            cluster_spec_arg = json.loads(cluster_spec)
        except ValueError as e:
            eprint("Invalid cluster spec JSON. Parse error: %s" % e)
            raise
    try:
        projects.run(
            uri,
            entry_point,
            version,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            parameters=param_dict,
            mode=mode,
            cluster_spec=cluster_spec_arg,
            git_username=git_username,
            git_password=git_password,
            use_conda=(not no_conda),
            storage_dir=storage_dir,
            block=mode == "local" or mode is None,
            run_id=run_id,
        )
    except projects.ExecutionException as e:
        _logger.error("=== %s ===", e)
        sys.exit(1)


@cli.command()
@click.option("--backend-store-uri", "--file-store", metavar="PATH",
              default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
              help="URI or path for backend store implementation. Acceptable backend store "
                   "are SQLAlchemy compatible implementation or local storage. "
                   "Example 'sqlite:///path/to/file.db'. "
                   "By default file backed store will be used. (default: ./mlruns).")
@click.option("--default-artifact-root", metavar="URI", default=None,
              help="Path to local directory to store artifacts, for new experiments. "
                   "Note that this flag does not impact already-created experiments. "
                   "Default: " + DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)
@click.option("--host", "-h", metavar="HOST", default="127.0.0.1",
              help="The network address to listen on (default: 127.0.0.1). "
                   "Use 0.0.0.0 to bind to all addresses if you want to access the UI from "
                   "other machines.")
@click.option("--port", "-p", default=5000,
              help="The port to listen on (default: 5000).")
@click.option("--gunicorn-opts", default=None,
              help="Additional command line options forwarded to gunicorn processes.")
def ui(backend_store_uri, default_artifact_root, host, port, gunicorn_opts):
    """
    Launch the MLflow tracking UI.

    The UI will be visible at http://localhost:5000 by default.
    """
    # Ensure that both backend_store_uri and default_artifact_uri are set correctly.
    if not backend_store_uri:
        backend_store_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    if not default_artifact_root:
        if _is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        else:
            default_artifact_root = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    # TODO: We eventually want to disable the write path in this version of the server.
    try:
        _run_server(backend_store_uri, default_artifact_root, host, port, 1, None, gunicorn_opts)
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
@click.option("--backend-store-uri", "--file-store", metavar="PATH",
              default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
              help="URI or path for backend store implementation. Acceptable backend store "
                   "are SQLAlchemy compatible implementation or local storage. Supports "
                   "various SQLAlchemy compatible database like SQLite, MySQL, PostgreSQL. As an "
                   "example MySQL backed store can be configured using connection string. "
                   "'mysql://<user_name>:<password>@<host>:<port>/<database_name>' "
                   "By default file based backed store will be used. (default: ./mlruns).")
@click.option("--default-artifact-root", metavar="URI", default=None,
              help="Local or S3 URI to store artifacts, for new experiments. "
                   "Note that this flag does not impact already-created experiments. "
                   "Default: Within file store")
@click.option("--host", "-h", metavar="HOST", default="127.0.0.1",
              help="The network address to listen on (default: 127.0.0.1). "
                   "Use 0.0.0.0 to bind to all addresses if you want to access the tracking "
                   "server from other machines.")
@click.option("--port", "-p", default=5000,
              help="The port to listen on (default: 5000).")
@click.option("--workers", "-w", default=4,
              help="Number of gunicorn worker processes to handle requests (default: 4).")
@click.option("--static-prefix", default=None, callback=_validate_static_prefix,
              help="A prefix which will be prepended to the path of all static paths.")
@click.option("--gunicorn-opts", default=None,
              help="Additional command line options forwarded to gunicorn processes.")
def server(backend_store_uri, default_artifact_root, host, port,
           workers, static_prefix, gunicorn_opts):
    """
    Run the MLflow tracking server.

    The server which listen on http://localhost:5000 by default, and only accept connections from
    the local machine. To let the server accept connections from other machines, you will need to
    pass --host 0.0.0.0 to listen on all network interfaces (or a specific interface address).
    """

    # Ensure that both backend_store_uri and default_artifact_uri are set correctly.
    if not backend_store_uri:
        backend_store_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH

    if not default_artifact_root:
        if _is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        else:
            eprint("Option 'default-artifact-root' is required, when backend store is not "
                   "local file based.")
            sys.exit(1)

    try:
        _run_server(backend_store_uri, default_artifact_root, host, port, workers, static_prefix,
                    gunicorn_opts)
    except ShellCommandException:
        eprint("Running the mlflow server failed. Please see the logs above for details.")
        sys.exit(1)


cli.add_command(mlflow.data.download)
cli.add_command(mlflow.pyfunc.cli.commands)
cli.add_command(mlflow.rfunc.cli.commands)
cli.add_command(mlflow.sagemaker.cli.commands)
cli.add_command(mlflow.experiments.commands)
cli.add_command(mlflow.store.cli.commands)
cli.add_command(mlflow.azureml.cli.commands)
cli.add_command(mlflow.runs.commands)

if __name__ == '__main__':
    cli()
