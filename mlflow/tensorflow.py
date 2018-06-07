"""Sample MLflow integration for Tensorflow."""

from __future__ import absolute_import

import json
import os
import dill as pickle

import click
import flask
import pandas
import sklearn
import tensorflow as tf

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


def save_tf_model(tf_model, path, conda_env=None, mlflow_model=Model()):
    """Save a Tensorflow model to a directory in the local file system."""
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_fn_file = os.path.join(path, "model_fn.pkl")
    model_dir_file = os.path.join(path, "model_dir.pkl")
    with open(model_fn_file, "wb") as out:
        pickle.dump(tf_model.model_fn, out)
    with open(model_dir_file, "wb") as out:
        pickle.dump(tf_model.model_dir, out)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow", data=path, env=conda_env)
    mlflow_model.add_flavor("tensorflow",
                            fn="model_fn.pkl",
                            dir="model_dir.pkl")
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_tf_model(tf_model, artifact_path):
    """Log a Tensorflow model as an MLflow artifact for the current run."""
    with TempDir() as tmp:
        local_path = tmp.path("model")
        run_id = mlflow.tracking.active_run().info.run_uuid
        mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
        save_tf_model(tf_model, local_path, mlflow_model=mlflow_model)
        mlflow.tracking.log_artifacts(local_path, artifact_path)


def _load_model_from_local_file(path):
    """Load a Tensorflow model saved as an MLflow artifact on the local file system."""
    model = Model.load(os.path.join(path, "MLmodel"))
    assert "tensorflow" in model.flavors
    params = model.flavors["tensorflow"]
    model_fn = None
    with open(os.path.join(path, params["fn"]), "rb") as f:
        model_fn = pickle.load(f)
    model_dir = None
    with open(os.path.join(path, params["dir"]), "rb") as f:
        model_dir = pickle.load(f)
    return tf.estimator.Estimator(model_fn, model_dir=model_dir)


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path, run_id=None):
    """Load a Tensorflow model from a local file (if run_id is None) or a run."""
    if run_id is not None:
        path = mlflow.tracking._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model_from_local_file(path)


@click.group("tensorflow")
def commands():
    """Serve Tensorflow models."""
    pass


@commands.command("serve")
@click.argument("model_path")
@click.option("--run_id", "-r", metavar="RUN_ID", help="Run ID to look for the model in.")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
def serve_model(model_path, run_id=None, port=None):
    """
    Serve a SciKit-Learn model saved with MLflow.

    If a run_id is specified, MODEL_PATH is treated as an artifact path within that run;
    otherwise it is treated as a local path.
    """
    model = load_model(run_id=run_id, path=model_path)
    app = flask.Flask(__name__)

    @app.route('/invocations', methods=['POST'])
    def predict():  # noqa
        if flask.request.content_type != 'application/json':
            return flask.Response(status=415, response='JSON data expected', mimetype='text/plain')
        data = flask.request.data.decode('utf-8')
        records = pandas.read_json(data, orient="records")
        predictions = model.predict(records)
        result = json.dumps({"predictions": predictions.tolist()})
        return flask.Response(status=200, response=result + "\n", mimetype='application/json')

    app.run(port=port)
