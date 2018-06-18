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

from mlflow.entities.experiment import Experiment
import mlflow.server as server
from mlflow.store import file_store
from mlflow.store.file_store import FileStore
from mlflow import tracking


@click.group()
@click.version_option()
def cli():
    pass


def _encode(string_val):
    if string_val is None:
        return string_val
    # In Python 3, strings are unicode values, so we just return
    if isinstance(string_val, str):
        return string_val
    # In Python 2: `encode` convert from unicode object -> UTF-8 encoded string
    return string_val.encode("utf-8")


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
# NOTE: github recommends using tokens as environment vars
# see https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/
@click.option("--git-username", metavar="USERNAME", envvar="MLFLOW_GIT_USERNAME",
              help="Username for HTTP(S) Git authentication. Only used when running on Databricks.")
@click.option("--git-password", metavar="PASSWORD", envvar="MLFLOW_GIT_PASSWORD",
              help="Password for HTTP(S) Git authentication. Only used when running on Databricks.")
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
        param_dict[_encode(name)] = _encode(value)
    try:
        projects.run(_encode(uri), _encode(entry_point), _encode(version),
                     experiment_id=experiment_id,
                     parameters=param_dict, mode=_encode(mode),
                     cluster_spec=_encode(cluster_spec),
                     git_username=_encode(git_username),
                     git_password=_encode(git_password), use_conda=(not no_conda),
                     use_temp_cwd=new_dir, storage_dir=_encode(storage_dir))
    except projects.ExecutionException as e:
        print(e.message, file=sys.stderr)
        sys.exit(1)


@cli.command()
@click.option("--file-store-path", default=None,
              help="The root of the backing file store for experiment and run data. Defaults to %s."
                   % file_store._default_root_dir())
@click.option("--host", default="127.0.0.1",
              help="The networking interface on which the UI server listens. Defaults to "
                   "127.0.0.1.  Use 0.0.0.0 to bind to all addresses, which is useful for running "
                   "inside of docker.")
def ui(file_store_path, host):
    """
    Run the MLflow tracking UI. The UI is served at http://localhost:5000.
    """
    server.handlers.store = FileStore(file_store_path)
    server.app.run(host)


cli.add_command(mlflow.sklearn.commands)
cli.add_command(mlflow.sklearn.commands)
cli.add_command(mlflow.data.download)
cli.add_command(mlflow.pyfunc.cli.commands)
cli.add_command(mlflow.sagemaker.cli.commands)
cli.add_command(mlflow.azureml.cli.commands)
cli.add_command(mlflow.experiments.commands)

if __name__ == '__main__':
    cli()
