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
import inspect
import gorilla

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params

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


@experimental
def autolog():
    """
    Enables automatic logging from XGBoost to MLflow.
    Logs paramters and metrics specified in the train function.
    Trained model is logged as an artifact to a 'model' directory.
    """
    import xgboost

    @gorilla.patch(xgboost)
    def train(*args, **kwargs):

        def mlflow_callback(env):
            """
            Callback for auto-logging metrics on each iteration.
            """
            try_mlflow_log(mlflow.log_metrics, dict(env.evaluation_result_list),
                           step=env.iteration)

        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        original = gorilla.get_original_attribute(xgboost, 'train')
        unlogged_params = ['dtrain', 'evals', 'obj', 'feval', 'evals_result',
                           'xgb_model', 'callbacks', 'learning_rates']
        log_fn_args_as_params(original, args, kwargs, unlogged_params)

        all_arg_names = inspect.getargspec(original)[0]
        num_pos_args = len(args)

        # checking if the 'callbacks' argument of train() is set.
        callbacks_index = all_arg_names.index('callbacks')
        if num_pos_args >= callbacks_index + 1:
            tmp_list = list(args)
            tmp_list[callbacks_index] += [mlflow_callback]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs and kwargs['callbacks'] is not None:
            kwargs['callbacks'] += [mlflow_callback]
        else:
            kwargs['callbacks'] = [mlflow_callback]

        model = original(*args, **kwargs)

        # checking if the 'early_stopping_rounds' argument of train() is set.
        early_stopping_index = all_arg_names.index('early_stopping_rounds')
        has_early_stopping = num_pos_args >= early_stopping_index + 1 or \
                             'early_stopping_rounds' in kwargs

        # if 'early_stopping_rounds' is set, the output model has
        # 'best_score', 'best_iteration', and 'best_ntree_limit'
        # even if early stopping didn't occur.
        if has_early_stopping:
            try_mlflow_log(mlflow.log_metric, 'best_score', model.best_score)
            try_mlflow_log(mlflow.log_metric, 'best_iteration', model.best_iteration)
            try_mlflow_log(mlflow.log_metric, 'best_ntree_limit', model.best_iteration)

        try_mlflow_log(log_model, model, artifact_path='model')

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)
        return model

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(xgboost, 'train', train, settings=settings))
