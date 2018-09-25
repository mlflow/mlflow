"""
The ``mlflow.keras`` module provides an API for logging and loading Keras models. This module
exports Keras models with the following flavors:

Keras (native) format
    This is the main flavor that can be loaded back into Keras.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os

import keras.backend as K
import pandas as pd

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


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
    >>> keras_model.compile(optimizer="rmsprop", loss="mse", metrics["accuracy"])
    >>> results = keras_model.fit(
    ...     x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
    ... # Save the model as an MLflow Model
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

    :param keras_model: Keras model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param kwargs: kwargs to pass to ``keras_model.save`` method.

    >>> from keras import Dense, layers
    >>> import mlflow
    >>> # Build, compile, and train your model
    >>> keras_model = ...
    >>> keras_model.compile(optimizer="rmsprop", loss="mse", metrics["accuracy"])
    >>> results = keras_model.fit(
    ...     x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
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
    def __init__(self, keras_model, graph, sess):
        self.keras_model = keras_model
        self._graph = graph
        self._sess = sess

    def predict(self, dataframe):
        with self._graph.as_default():
            with self._sess.as_default():
                predicted = pd.DataFrame(self.keras_model.predict(dataframe))
        predicted.index = dataframe.index
        return predicted


def _load_pyfunc(model_file):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    if K._BACKEND == 'tensorflow':
        import tensorflow as tf
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        # By default tf backed models depend on the global graph and session.
        # We create an use new Graph and Session and store them with the model
        # This way the model is independent on the global state.
        with graph.as_default():
            with sess.as_default():  # pylint:disable=not-context-manager
                K.set_learning_phase(0)
                m = _load_model(model_file)
        return _KerasModelWrapper(m, graph, sess)
    else:
        raise Exception("Unsupported backend '%s'" % K._BACKEND)


def load_model(path, run_id=None):
    """
    Load a Keras model from a local file (if ``run_id`` is None) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.keras.log_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.

    >>> # Load persisted model as a Keras model or as a PyFunc, call predict() on a Pandas DataFrame
    >>> keras_model = mlflow.keras.load_model("models", run_id="96771d893a5e46159d9f3b49bf9013e2")
    >>> predictions = keras_model.predict(x_test)
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model(os.path.join(path, "model.h5"))
