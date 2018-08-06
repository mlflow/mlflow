from __future__ import absolute_import

import sys
import os
import subprocess

import click
import pandas

from mlflow.pyfunc import load_pyfunc, scoring_server, _load_model_env
from mlflow.tracking import _get_model_log_dir
from mlflow.utils import cli_args
from mlflow.projects import _maybe_create_conda_env, _get_conda_env_name, _get_conda_bin_executable
from mlflow.utils.logging_utils import eprint

import six


def _rerun_in_conda(conda_env_path):
    """ Rerun CLI command inside a to-be-created conda environment."""
    conda_env_name = _maybe_create_conda_env(conda_env_path)
    activate_path = _get_conda_bin_executable("activate")
    commands = []
    commands.append("source {} {}".format(activate_path, conda_env_name))
    commands.append(_reconstruct_command_line())
    commandline = " && ".join(commands)
    child = subprocess.Popen(["bash", "-c", commandline], close_fds=True)
    exit_code = child.wait()
    return exit_code


def _reconstruct_command_line():
    """Reconstruct original command line from click context"""
    ctx = click.get_current_context()
    cmdline = ctx.command_path
    if not cmdline.startswith("mlflow"):
        cmdline = 'mlflow ' + cmdline
    params = ctx.params.copy()
    # enforce newcommand to contaon --no-conda so we don't run into conda activation loop
    params['no_conda'] = True
    for key, val in params.items():
        opt = " --" + key.replace('_', '-')
        if isinstance(val, bool):
            if val:
                cmdline += opt
        elif isinstance(val, six.string_types) or isinstance(val, (int, float)):
            cmdline += "{} {}".format(opt, val)
        elif val is None:
            pass
        else:
            raise NotImplementedError('unknown {} arg {} {}'.format(type(val), opt, val))

    return cmdline


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
