from __future__ import absolute_import

import sys

import click
import pandas

from mlflow.pyfunc import load_pyfunc, scoring_server
from mlflow.tracking import _get_model_log_dir
from mlflow.utils import cli_args


@click.group("pyfunc")
def commands():
    """Serve Python models locally."""
    pass


@commands.command("serve")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
def serve(model_path, run_id, port):
    """
    Serve a PythonFunction model saved with MLflow.

    If a ``run_id`` is specified, ``model-path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    app = scoring_server.init(load_pyfunc(model_path))
    app.run(port=port)


@commands.command("predict")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--input-path", "-i", help="CSV containing pandas DataFrame to predict against.",
              required=True)
@click.option("--output-path", "-o", help="File to output results to as CSV file." +
                                          " If not provided, output to stdout.")
def predict(model_path, run_id, input_path, output_path):
    """
    Load a pandas DataFrame and runs a PythonFunction model saved with MLflow against it.
    Return the prediction results as a CSV-formatted pandas DataFrame.

    If a ``run-id`` is specified, ``model-path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    model = load_pyfunc(model_path)
    df = pandas.read_csv(input_path)
    result = model.predict(df)
    out_stream = sys.stdout
    if output_path:
        out_stream = open(output_path, 'w')
    pandas.DataFrame(data=result).to_csv(out_stream, header=False, index=False)
