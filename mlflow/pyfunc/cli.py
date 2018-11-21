from __future__ import absolute_import

import os
from six.moves import shlex_quote
import subprocess
import sys
import logging

import click
import pandas

from mlflow.projects import _get_conda_bin_executable, _get_or_create_conda_env
from mlflow.pyfunc import load_pyfunc, scoring_server, _load_model_env
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils import cli_args


_logger = logging.getLogger(__name__)


def _rerun_in_conda(conda_env_path):
    """ Rerun CLI command inside a to-be-created conda environment."""
    conda_env_name = _get_or_create_conda_env(conda_env_path)
    activate_path = _get_conda_bin_executable("activate")
    commands = []
    commands.append("source {} {}".format(activate_path, conda_env_name))
    safe_argv = [shlex_quote(arg) for arg in sys.argv]
    commands.append(" ".join(safe_argv) + " --no-conda")
    commandline = " && ".join(commands)
    _logger.info("=== Running command '%s'", commandline)
    child = subprocess.Popen(["bash", "-c", commandline], close_fds=True)
    exit_code = child.wait()
    return exit_code


@click.group("pyfunc")
def commands():
    """
    Serve Python models locally.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("serve")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--host", "-h", default="127.0.0.1", help="Server host. [default: 127.0.0.1]")
@cli_args.NO_CONDA
def serve(model_path, run_id, port, host, no_conda):
    """
    Serve a pyfunc model saved with MLflow by launching a webserver on the specified
    host and port. For information about the input data formats accepted by the webserver,
    see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment.

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
    app.run(port=port, host=host)


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
    Load a pandas DataFrame and runs a python_function model saved with MLflow against it.
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
