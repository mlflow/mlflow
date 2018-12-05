from __future__ import absolute_import

import click
import os
import subprocess
import logging

from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils import cli_args


_logger = logging.getLogger(__name__)


@click.group("rfunc")
def commands():
    """
    Serve R models locally.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


def execute(command):
    _logger.info("=== Rscript -e %s) ===", command)
    env = os.environ.copy()
    process = subprocess.Popen(["Rscript", "-e", command], close_fds=True, env=env)
    process.wait()


def str_optional(s):
    if s is None:
        return ''
    return str(s)


@commands.command("serve")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
def serve(model_path, run_id, port):
    """
    Serve an RFunction model saved with MLflow.

    If a ``run_id`` is specified, ``model-path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)

    command = "mlflow::mlflow_rfunc_serve('{0}', port = {1})".format(model_path, port)
    execute(command)


@commands.command("predict")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--input-path", "-i", help="JSON or CSV containing DataFrame to predict against.",
              required=True)
@click.option("--output-path", "-o", help="File to output results to as JSON or CSV file." +
                                          " If not provided, output to stdout.")
def predict(model_path, run_id, input_path, output_path):
    """
    Serve an RFunction model saved with MLflow.
    Return the prediction results as a JSON DataFrame.

    If a ``run-id`` is specified, ``model-path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)

    str_cmd = "mlflow::mlflow_rfunc_predict('{0}', '{1}', '{2}')"
    command = str_cmd.format(model_path, input_path, str_optional(output_path))

    execute(command)
