"""
The ``mlflow.xgboost`` module provides an API for logging and loading XGBoost models.
This module exports XGBoost models with the following flavors:

XGBoost (native) format
    This is the main flavor that can be loaded back into XGBoost.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _xgboost.Booster:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster
.. _xgboost.Booster.save_model:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.save_model
.. _scikit-learn API:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
"""

from __future__ import absolute_import

import os
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException

FLAVOR_NAME = "xgboost"


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import xgboost as xgb

    return _mlflow_conda_env(
        additional_conda_deps=None,
        # XGBoost is not yet available via the default conda channels, so we install it via pip
        additional_pip_deps=[
            "xgboost=={}".format(xgb.__version__),
        ],
        additional_conda_channels=None)


def save_model(xgb_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save an XGBoost model to a path on the local file system.

    :param xgb_model: XGBoost model (an instance of `xgboost.Booster`_) to be saved.
                      Note that models that implement the `scikit-learn API`_  are not supported.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'xgboost==0.90'
                                ]
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    """
    import xgboost as xgb

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.xgb"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)

    # Save an XGBoost model
    xgb_model.save_model(model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.xgboost",
                        data=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, xgb_version=xgb.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(xgb_model, artifact_path, conda_env=None, registered_model_name=None, **kwargs):
    """
    Log an XGBoost model as an MLflow artifact for the current run.

    :param xgb_model: XGBoost model (an instance of `xgboost.Booster`_) to be saved.
                      Note that models that implement the `scikit-learn API`_  are not supported.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'xgboost==0.90'
                                ]
                            ]
                        }
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param kwargs: kwargs to pass to `xgboost.Booster.save_model`_ method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.xgboost,
              registered_model_name=registered_model_name,
              xgb_model=xgb_model, conda_env=conda_env, **kwargs)


def _load_model(path):
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(os.path.abspath(path))
    return model


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``xgboost`` flavor.
    """
    return _XGBModelWrapper(_load_model(path))


def load_model(model_uri):
    """
    Load an XGBoost model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: An XGBoost model (an instance of `xgboost.Booster`_)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    xgb_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.xgb"))
    return _load_model(path=xgb_model_file_path)


class _XGBModelWrapper:
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model

    def predict(self, dataframe):
        import xgboost as xgb
        return self.xgb_model.predict(xgb.DMatrix(dataframe))
