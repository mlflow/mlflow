"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <http://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import json
import os
import pickle
import shutil

import click
import flask
import pandas
import sklearn

from mlflow.utils import cli_args
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
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    >>> import mlflow.sklearn
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> sk_model = tree.DecisionTreeClassifier()
    >>> sk_model = sk_model.fit(iris.data, iris.target)
    >>> #set path to location for persistence
    >>> sk_path_dir = ...
    >>> mlflow.sklearn.save_model(sk_model, sk_path_dir)
    """
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_file = os.path.join(path, "model.pkl")
    with open(model_file, "wb") as out:
        pickle.dump(sk_model, out)
    model_conda_env = None
    if conda_env:
        model_conda_env = os.path.basename(os.path.abspath(conda_env))
        shutil.copyfile(conda_env, os.path.join(path, model_conda_env))
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn", data="model.pkl",
                        env=model_conda_env)
    mlflow_model.add_flavor("sklearn",
                            pickled_model="model.pkl",
                            sklearn_version=sklearn.__version__)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(sk_model, artifact_path, conda_env=None):
    """
    Log a scikit-learn model as an MLflow artifact for the current run.

    :param sk_model: scikit-learn model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, this decribes the environment
           this model should be run in. At minimum, it should specify python, scikit-learn,
           and mlflow with appropriate versions.

    >>> import mlflow
    >>> import mlflow.sklearn
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> sk_model = tree.DecisionTreeClassifier()
    >>> sk_model = sk_model.fit(iris.data, iris.target)
    >>> #set the artifact_path to location where experiment artifacts will be saved
    >>> #log model params
    >>> mlflow.log_param("criterion", sk_model.criterion)
    >>> mlflow.log_param("splitter", sk_model.splitter)
    >>> #log model
    >>> mlflow.sklearn.log_model(sk_model, "sk_models")
    """
    return Model.log(artifact_path=artifact_path,
                     flavor=mlflow.sklearn,
                     sk_model=sk_model,
                     conda_env=conda_env)


def _load_model_from_local_file(path):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system."""
    # TODO: we could validate the SciKit-Learn version here
    model = Model.load(os.path.join(path, "MLmodel"))
    assert "sklearn" in model.flavors
    params = model.flavors["sklearn"]
    with open(os.path.join(path, params["pickled_model"]), "rb") as f:
        return pickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path, run_id=None):
    """
    Load a scikit-learn model from a local file (if ``run_id`` is None) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.sklearn.save_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.

    >>> import mlflow.sklearn
    >>> sk_model = mlflow.sklearn.load_model("sk_models", run_id="96771d893a5e46159d9f3b49bf9013e2")
    >>> #use Pandas DataFrame to make predictions
    >>> pandas_df = ...
    >>> predictions = sk_model.predict(pandas_df)
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model_from_local_file(path)


@click.group("sklearn")
def commands():
    """
    Serve scikit-learn models locally.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("serve")
@cli_args.MODEL_PATH
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
