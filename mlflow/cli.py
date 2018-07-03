from __future__ import print_function

import sys

import click

import mlflow.azureml.cli
import mlflow.projects as projects
import mlflow.sklearn
import mlflow.data
import mlflow.experiments
import mlflow.pyfunc.cli
import mlflow.sagemaker.cli
import mlflow.server

from mlflow.entities.experiment import Experiment
from mlflow import tracking


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
@click.option("--experiment-id", envvar=tracking._EXPERIMENT_ID_ENV_VAR, type=click.INT,
              help="ID of the experiment under which to launch the run. Defaults to %s" %
                   Experiment.DEFAULT_EXPERIMENT_ID)
# TODO: Add tracking server argument once we have it working.
@click.option("--mode", "-m", metavar="MODE",
              help="Execution mode to use for run. Supported values: 'local' (runs project"
                   "locally) and 'databricks' (runs project on a Databricks cluster)."
                   "Defaults to 'local'. If running against Databricks, will run against the "
                   "Databricks workspace specified in the default Databricks CLI profile. "
                   "See https://github.com/databricks/databricks-cli for more info on configuring "
                   "a Databricks CLI profile.")
@click.option("--cluster-spec", metavar="FILE",
              help="Path to JSON file describing the cluster to use when launching a run on "
                   "Databricks. See "
                   "https://docs.databricks.com/api/latest/jobs.html#jobsclusterspecnewcluster for "
                   "more info. Note that MLflow runs are currently launched against a new cluster.")
@click.option("--git-username", metavar="USERNAME", envvar="MLFLOW_GIT_USERNAME",
              help="Username for HTTP(S) Git authentication.")
@click.option("--git-password", metavar="PASSWORD", envvar="MLFLOW_GIT_PASSWORD",
              help="Password for HTTP(S) Git authentication.")
@click.option("--no-conda", is_flag=True,
              help="If specified, will assume that MLflow is running within a Conda environment "
                   "with the necessary dependencies for the current project instead of attempting "
                   "to create a new conda environment. Only valid if running locally.")
@click.option("--new-dir", is_flag=True,
              help="Only valid when `mode` is 'local' and `uri` points to a local directory."
                   "If specified, copies the project into a temporary working directory and "
                   "runs it from there. Otherwise, uses `uri` as the working directory when "
                   "running the project. Note that Git projects are always run from a temporary "
                   "working directory.")
@click.option("--storage-dir", envvar="MLFLOW_TMP_DIR",
              help="Only valid when `mode` is local."
                   "MLflow will download artifacts from distributed URIs passed to parameters of "
                   "type 'path' to subdirectories of storage_dir.")
def run(uri, entry_point, version, param_list, experiment_id, mode, cluster_spec, git_username,
        git_password, no_conda, new_dir, storage_dir):
    """
    Run an MLflow project from the given URI.

    If running locally (the default), the URI can be either a Git repository URI or a local path.
    If running on Databricks, the URI must be a Git repository.

    By default, Git projects will run in a new working directory with the given parameters, while
    local projects will run from the project's root directory.
    """
    param_dict = {}
    for s in param_list:
        index = s.find("=")
        if index == -1:
            print("Invalid format for -P parameter: '%s'. Use -P name=value." % s, file=sys.stderr)
            sys.exit(1)
        name = s[:index]
        value = s[index + 1:]
        if name in param_dict:
            print("Repeated parameter: '%s'" % name, file=sys.stderr)
            sys.exit(1)
        param_dict[name] = value
    try:
        projects.run(uri,
                     entry_point,
                     version,
                     experiment_id=experiment_id,
                     parameters=param_dict,
                     mode=mode,
                     cluster_spec=cluster_spec,
                     git_username=git_username,
                     git_password=git_password,
                     use_conda=(not no_conda),
                     use_temp_cwd=new_dir,
                     storage_dir=storage_dir)
    except projects.ExecutionException as e:
        print(e.message, file=sys.stderr)
        sys.exit(1)


@cli.command()
@click.option("--file-store", metavar="PATH", default=None,
              help="The root of the backing file store for experiment and run data "
                   "(default: ./mlruns).")
@click.option("--host", "-h", metavar="HOST", default="127.0.0.1",
              help="The network address to listen on (default: 127.0.0.1). "
                   "Use 0.0.0.0 to bind to all addresses if you want to access the UI from "
                   "other machines.")
@click.option("--port", "-p", default=5000,
              help="The port to listen on (default: 5000).")
def ui(file_store, host, port):
    """
    Launch the MLflow tracking UI.

    The UI will be visible at http://localhost:5000 by default.
    """
    # TODO: We eventually want to disable the write path in this version of the server.
    mlflow.server._run_server(file_store, file_store, host, port, 1)


@cli.command()
@click.option("--file-store", metavar="PATH", default=None,
              help="The root of the backing file store for experiment and run data "
                   "(default: ./mlruns).")
@click.option("--artifact-root", metavar="URI", default=None,
              help="Local or S3 URI to store artifacts in (default: inside file store).")
@click.option("--host", "-h", metavar="HOST", default="127.0.0.1",
              help="The network address to listen on (default: 127.0.0.1). "
                   "Use 0.0.0.0 to bind to all addresses if you want to access the tracking "
                   "server from other machines.")
@click.option("--port", "-p", default=5000,
              help="The port to listen on (default: 5000).")
@click.option("--workers", "-w", default=4,
              help="Number of gunicorn worker processes to handle requests (default: 4).")
def server(file_store, artifact_root, host, port, workers):
    """
    Run the MLflow tracking server.

    The server which listen on http://localhost:5000 by default, and only accept connections from
    the local machine. To let the server accept connections from other machines, you will need to
    pass --host 0.0.0.0 to listen on all network interfaces (or a specific interface address).
    """
    mlflow.server._run_server(file_store, artifact_root, host, port, workers)


cli.add_command(mlflow.sklearn.commands)
cli.add_command(mlflow.data.download)
cli.add_command(mlflow.pyfunc.cli.commands)
cli.add_command(mlflow.sagemaker.cli.commands)
cli.add_command(mlflow.azureml.cli.commands)
cli.add_command(mlflow.experiments.commands)

if __name__ == '__main__':
    cli()
