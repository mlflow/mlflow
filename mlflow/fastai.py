"""
The ``mlflow.fastai`` module provides an API for logging and loading fast.ai models. This module
exports fast.ai models with the following flavors:

fastai (native) format
    This is the main flavor that can be loaded back into fastai.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _fastai.Learner:
    https://docs.fast.ai/basic_train.html#Learner
.. _fastai.Learner.export:
    https://docs.fast.ai/basic_train.html#Learner.export
"""

from __future__ import absolute_import

import os
import yaml
import pandas as pd
import numpy as np

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

from fastai.tabular import TabularList
from fastai.basic_data import DatasetType


FLAVOR_NAME = "fastai"


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import fastai
    pip_deps = None
    if include_cloudpickle:
        import cloudpickle
        pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]
    return _mlflow_conda_env(
        additional_conda_deps=[
            "fastai={}".format(fastai.__version__),
        ],
        additional_pip_deps=pip_deps,
        additional_conda_channels=None
    )


def save_model(fastai_learner, path, conda_env=None, mlflow_model=Model(), **kwargs):
    """
    Save a fastai Learner to a path on the local file system.

    :param fastai_learner: fastai Learner to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the
                      dependencies contained in :func:`get_default_conda_env()`. If
                      ``None``, the default :func:`get_default_conda_env()` environment is
                      added to the model. The following is an *example* dictionary
                      representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'fastai=1.0.60',
                            ]
                        }
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param kwargs: kwargs to pass to ``Learner.save`` method.
    """
    import fastai
    from pathlib import Path

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.fastai"
    model_data_path = os.path.join(path, model_data_subpath)
    model_data_path = Path(model_data_path)
    os.makedirs(path)

    # Save an Learner
    fastai_learner.export(model_data_path, **kwargs)

    conda_env_subpath = "conda.yaml"

    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.fastai",
                        data=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, fastai_version=fastai.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(fastai_learner, artifact_path, conda_env=None, registered_model_name=None, **kwargs):
    """
    Log a fastai model as an MLflow artifact for the current run.

    :param fastai_learner: Fastai model (an instance of `fastai.Learner`_) to be saved.
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
                                'fastai=1.0.60',
                            ]
                        }
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param kwargs: kwargs to pass to `fastai.Learner.export`_ method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.fastai,
              registered_model_name=registered_model_name,
              fastai_learner=fastai_learner, conda_env=conda_env, **kwargs)


def _load_model(path):
    from fastai.basic_train import load_learner
    abspath = os.path.abspath(path)
    path, file = os.path.split(abspath)
    return load_learner(path, file)


class _FastaiModelWrapper:
    def __init__(self, learner):
        self.learner = learner

    def predict(self, dataframe):
        test_data = TabularList.from_df(dataframe, cont_names=self.learner.data.cont_names)
        self.learner.data.add_test(test_data)
        preds, target = self.learner.get_preds(DatasetType.Test)
        preds = pd.Series(map(np.array, preds.numpy()), name='predictions')
        target = pd.Series(target.numpy(), name='target')
        return pd.concat([preds, target], axis='columns')


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``fastai`` flavor.
    """
    return _FastaiModelWrapper(_load_model(path))


def load_model(model_uri):
    """
    Load a fastai model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A fastai model (an instance of `fastai.Learner`_).
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.fastai"))
    return _load_model(path=model_file_path)
