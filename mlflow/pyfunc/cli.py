
from __future__ import absolute_import

import sys

import click
import pandas


from mlflow.pyfunc import load_pyfunc, scoring_server
from mlflow.tracking import _get_model_log_dir


@click.group("pyfunc")
def commands():
    """Serve Python models locally."""
    pass


@commands.command("serve")
@click.option("--model-path", "-m,", help="model path", required=True)
@click.option("--run_id", "-r", default=None, help="run_id")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
def serve(model_path, run_id, port):
    """
    Serve a PythonFunction model saved with MLflow.

    If a run_id is specified, MODEL_PATH is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    app = scoring_server.init(load_pyfunc(model_path))
    app.run(port=port)


@commands.command("predict")
@click.option("--model-path", "-m,", help="model path", required=True)
@click.option("--run_id", "-r", default=None, help="run_id")
@click.option("--input-path", "-i", help="CSV containing Pandas DataFrame to predict against",
              required=True)
@click.option("--output-path", "-o", help="Results will be output as CSV to this file." +
              " If not provided, stdout will be used.")
def predict(model_path, run_id, input_path, output_path):
    """
    Loads a Pandas DataFrame and runs a PythonFunction model saved with MLflow against it.
    This method will return the prediction results as a CSV-formatted Pandas DataFrame.

    If a run_id is specified, MODEL_PATH is treated as an artifact path within that run;
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
    pandas.DataFrame({'results': result}).to_csv(out_stream, header=True, index=False)
