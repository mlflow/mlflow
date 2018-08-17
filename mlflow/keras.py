"""MLflow integration for Keras."""

from __future__ import absolute_import

import os

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking

import pandas as pd


def save_model(keras_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save a Keras model to a path on the local file system.

    :param keras_model: Keras model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    """
    import keras

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_file = os.path.join(path, "model.h5")
    keras_model.save(model_file)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.keras",
                        data="model.h5", env=conda_env)
    mlflow_model.add_flavor("keras", keras_version=keras.__version__)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(keras_model, artifact_path, **kwargs):
    """Log a Keras model as an MLflow artifact for the current run."""
    Model.log(artifact_path=artifact_path, flavor=mlflow.keras,
              keras_model=keras_model, **kwargs)


def _load_model(model_file):
    import keras.models
    return keras.models.load_model(os.path.abspath(model_file))


class _KerasModelWrapper:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def predict(self, dataframe):
        predicted = pd.DataFrame(self.keras_model.predict(dataframe))
        predicted.index = dataframe.index
        return predicted


def load_pyfunc(model_file):
    """
    Loads a Keras model as a PyFunc from the passed-in persisted Keras model file.

    :param model_file: Path to Keras model file.
    :return: PyFunc model.
    """
    return _KerasModelWrapper(_load_model(model_file))


def load_model(path, run_id=None):
    """
    Load a Keras model from a local file (if run_id is None) or a run.
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model(os.path.join(path, "model.h5"))
