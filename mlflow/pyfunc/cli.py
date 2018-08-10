from __future__ import absolute_import

import sys
import os
import subprocess
import hashlib
import json

import click
import pandas
import six

from mlflow.pyfunc import load_pyfunc, scoring_server, _load_model_env
from mlflow.tracking import _get_model_log_dir
from mlflow.utils import cli_args, process
from mlflow.utils.logging_utils import eprint
from mlflow.projects import _get_conda_bin_executable


def _get_conda_env_name(conda_env_path):
    with open(conda_env_path) as conda_env_file:
        conda_env_hash = hashlib.sha1(conda_env_file.read().encode("utf-8")).hexdigest()
    return "mlflow-%s" % conda_env_hash


def _maybe_create_conda_env(conda_env_path):
    conda_env = _get_conda_env_name(conda_env_path)
    conda_path = _get_conda_bin_executable("conda")
    try:
        process.exec_cmd([conda_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException("Could not find conda executable at {0}. "
                                 "Please ensure conda is installed as per the instructions "
                                 "at https://conda.io/docs/user-guide/install/index.html. You may "
                                 "also configure MLflow to look for a specific conda executable "
                                 "by setting the {1} environment variable to the path of the conda "
                                 "executable".format(conda_path, MLFLOW_CONDA))
    (_, stdout, _) = process.exec_cmd([conda_path, "env", "list", "--json"])
    env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]

    conda_action = 'create'
    if conda_env not in env_names:
        eprint('=== Creating conda environment %s ===' % conda_env)
        process.exec_cmd([conda_path, "env", conda_action, "-n", conda_env, "--file",
                          conda_env_path], stream_output=True)
    return conda_env


def _rerun_in_conda(conda_env_path):
    """ Rerun CLI command inside a to-be-created conda environment."""
    conda_env_name = _maybe_create_conda_env(conda_env_path)
    activate_path = _get_conda_bin_executable("activate")
    commands = []
    commands.append("source {} {}".format(activate_path, conda_env_name))
    commands.append(" ".join(sys.argv) + " --no-conda")
    commandline = " && ".join(commands)
    print("X",commandline)
    eprint("=== Running command '{}'".format(commandline))
    child = subprocess.Popen(["bash", "-c", commandline], close_fds=True)
    exit_code = child.wait()
    return exit_code


@click.group("pyfunc")
def commands():
    """Serve Python models locally."""
    pass


@commands.command("serve")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@cli_args.NO_CONDA
def serve(model_path, run_id, port, no_conda):
    """
    Serve a PythonFunction model saved with MLflow.

    If a ``run_id`` is specified, ``model-path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)

    model_env_file = _load_model_env(model_path)
    if not no_conda and model_env_file is not None:
        conda_env_path = os.path.join(model_path, model_env_file)
        return _rerun_in_conda(conda_env_path)

    app = scoring_server.init(load_pyfunc(model_path))
    app.run(port=port)


@commands.command("predict")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--input-path", "-i", help="CSV containing pandas DataFrame to predict against.",
              required=True)
@click.option("--output-path", "-o", help="File to output results to as CSV file." +
                                          " If not provided, output to stdout.")
@cli_args.NO_CONDA
def predict(model_path, run_id, input_path, output_path, no_conda):
    """
    Load a pandas DataFrame and runs a PythonFunction model saved with MLflow against it.
    Return the prediction results as a CSV-formatted pandas DataFrame.

    If a ``run-id`` is specified, ``model-path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)

    model_env_file = _load_model_env(model_path)
    if not no_conda and model_env_file is not None:
        conda_env_path = os.path.join(model_path, model_env_file)
        return _rerun_in_conda(conda_env_path)

    model = load_pyfunc(model_path)
    df = pandas.read_csv(input_path)
    result = model.predict(df)
    out_stream = sys.stdout
    if output_path:
        out_stream = open(output_path, 'w')
    pandas.DataFrame(data=result).to_csv(out_stream, header=False, index=False)
