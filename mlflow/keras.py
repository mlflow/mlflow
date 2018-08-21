"""Sample MLflow integration for Keras."""

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

    >>> import mlflow
    >>> # Build, compile, and train your model
    >>> keras_model = ...
    >>> keras_model_path = ...
    >>> keras_model.compile(optimizer="rmsprop",loss="mse", metrics["accuracy"])
    >>> results = keras_model.fit(x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
    ... # save the model in h5 format
    >>> mlflow.keras.save_model(keras_model, keras_model_path)
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
    """
    Log a Keras model as an MLflow artifact for the current run.

    :param keras_model: Keras model type to be logged as an artifact
    :param artifact_path: path or directory name under artifacts. Usually, "models" is a good choice
    :param kwargs:

    >>> from keras import Dense, layers
    >>> import mlflow
    >>> # Build, compile, and train your model
    >>> keras_model = ...
    >>> keras_model.compile(optimizer="rmsprop", loss="mse", metrics["accuracy"])
    >>> results = keras_model.fit(x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
    ...
    >>> # Log metrics and log the model
    >>> with mlflow.start_run() as run:
    >>>   mlflow.keras.log_model(keras_model, "models")
    """

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
    Load a Python Function model from a local file.

    :param model_file: path from where to load
    :return: The model as PyFunc.

    >>> model_file = "/tmp/pyfunc-keras-model"
    >>> keras_model = mlflow.keras.load_pyfunc(mode_file)
    >>> predictions = keras_mode.predict(x_test)
    """

    return _KerasModelWrapper(_load_model(model_file))


def load_model(path, run_id=None):
    """
    Load a Keras model from a local file (if run_id is None) or a run.
    
    :param path: artifact path
    :param run_id: run_id of a particular run
    :return: Keras model

    >>> #Load the model as Keras model or as pyfunc and use its predict() method
    >>> keras_model = mlflow.keras.load_model("models", run_id="96771d893a5e46159d9f3b49bf9013e2")
    >>> predictions = keras_model.predict(x_test)
    """

    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model(os.path.join(path, "model.h5"))
