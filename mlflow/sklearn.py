"""MLflow integration for scikit-learn."""

from __future__ import absolute_import

import json
import os
import pickle

import click
import flask
import pandas
import sklearn

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


def save_model(sk_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save a scikit-learn model to a path on the local file system.

    :param sk_model: scikit-learn model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Path to a Conda environment file. If provided, this decribes the environment
           this model should be run in. At minimum, it should specify python, scikit-learn,
           and mlflow with appropriate versions.
    :param mlflow_model: MLflow model config this flavor is being added to.
    """
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_file = os.path.join(path, "model.pkl")
    with open(model_file, "wb") as out:
        pickle.dump(sk_model, out)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn", data="model.pkl",
                        env=conda_env)
    mlflow_model.add_flavor("sklearn",
                            pickled_model="model.pkl",
                            sklearn_version=sklearn.__version__)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(sk_model, artifact_path):
    """Log a scikit-learn model as an MLflow artifact for the current run."""

    with TempDir() as tmp:
        local_path = tmp.path("model")
        # TODO: I get active_run_id here but mlflow.tracking.log_output_files has its own way
        run_id = mlflow.tracking._get_or_start_run().run_info.run_uuid
        mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
        save_model(sk_model, local_path, mlflow_model=mlflow_model)
        mlflow.tracking.log_artifacts(local_path, artifact_path)


def _load_model_from_local_file(path):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system."""
    # TODO: we could validate the SciKit-Learn version here
    model = Model.load(os.path.join(path, "MLmodel"))
    assert "sklearn" in model.flavors
    params = model.flavors["sklearn"]
    with open(os.path.join(path, params["pickled_model"]), "rb") as f:
        return pickle.load(f)


def load_pyfunc(path):
    """Load a Python Function model from a local file."""

    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path, run_id=None):
    """Load a scikit-learn model from a local file (if ``run_id`` is None) or a run."""
    if run_id is not None:
        path = mlflow.tracking._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model_from_local_file(path)


@click.group("sklearn")
def commands():
    """Serve scikit-learn models."""
    pass


@commands.command("serve")
@click.argument("model_path")
@click.option("--run_id", "-r", metavar="RUN_ID", help="Run ID to look for the model in.")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--host", default="127.0.0.1",
              help="The networking interface on which the prediction server listens. Defaults to "
                   "127.0.0.1.  Use 0.0.0.0 to bind to all addresses, which is useful for running "
                   "inside of docker.")
def serve_model(model_path, run_id=None, port=None, host="127.0.0.1"):
    """
    Serve a scikit-learn model saved with MLflow.

    If ``run_id`` is specified, ``model_path`` is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    model = load_model(run_id=run_id, path=model_path)
    app = flask.Flask(__name__)

    @app.route('/invocations', methods=['POST'])
    def predict():  # pylint: disable=unused-variable
        if flask.request.content_type != 'application/json':
            return flask.Response(status=415, response='JSON data expected', mimetype='text/plain')
        data = flask.request.data.decode('utf-8')
        records = pandas.read_json(data, orient="records")
        predictions = model.predict(records)
        result = json.dumps({"predictions": predictions.tolist()})
        return flask.Response(status=200, response=result + "\n", mimetype='application/json')

    app.run(host, port=port)
