"""
The ``mlflow.h2o`` module provides an API for logging and loading H2O models. This module exports
H2O models with the following flavors:

H20 (native) format
    This is the main flavor that can be loaded back into H2O.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import yaml

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


def save_model(h2o_model, path, conda_env=None, mlflow_model=Model(), settings=None):
    """
    Save an H2O model to a path on the local file system.

    :param h2o_model: H2O model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    """
    import h2o

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    model_dir = os.path.join(path, "model.h2o")
    os.makedirs(model_dir)

    # Save h2o-model
    h2o_save_location = h2o.save_model(model=h2o_model, path=model_dir, force=True)
    model_file = os.path.basename(h2o_save_location)

    # Save h2o-settings
    if settings is None:
        settings = {}
    settings['full_file'] = h2o_save_location
    settings['model_file'] = model_file
    settings['model_dir'] = model_dir
    with open(os.path.join(model_dir, "h2o.yaml"), 'w') as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.h2o",
                        data="model.h2o", env=conda_env)
    mlflow_model.add_flavor("h2o", saved_model=model_file, h2o_version=h2o.__version__)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(h2o_model, artifact_path, **kwargs):
    """
    Log an H2O model as an MLflow artifact for the current run.

    :param h2o_model: H2O model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param kwargs: kwargs to pass to ``h2o.save_model`` method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.h2o,
              h2o_model=h2o_model, **kwargs)


def _load_model(path, init=False):
    import h2o
    path = os.path.abspath(path)
    with open(os.path.join(path, "h2o.yaml")) as f:
        params = yaml.safe_load(f.read())
    if init:
        h2o.init(**(params["init"] if "init" in params else {}))
        h2o.no_progress()
    return h2o.load_model(os.path.join(path, params['model_file']))


class _H2OModelWrapper:
    def __init__(self, h2o_model):
        self.h2o_model = h2o_model

    def predict(self, dataframe):
        import h2o
        predicted = self.h2o_model.predict(h2o.H2OFrame(dataframe)).as_data_frame()
        predicted.index = dataframe.index
        return predicted


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _H2OModelWrapper(_load_model(path, init=True))


def load_model(path, run_id=None):
    """
    Load an H2O model from a local file (if ``run_id`` is ``None``) or a run.
    This function expects there is an H2O instance initialised with ``h2o.init``.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.h2o.save_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model(os.path.join(path, "model.h2o"))
