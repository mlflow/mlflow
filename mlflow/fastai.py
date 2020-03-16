"""
The ``mlflow.fastai`` module provides an API for logging and loading fast.ai models. This module
exports fast.ai models with the following flavors:

Keras (native) format
    This is the main flavor that can be loaded back into Keras.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import importlib
import os
import yaml
import gorilla
import tempfile
import shutil

import pandas as pd

from distutils.version import LooseVersion
from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params


FLAVOR_NAME = "fastai"
_MODEL_SAVE_PATH = "model.h5"
# Conda env subpath when saving/loading model
_CONDA_ENV_SUBPATH = "conda.yaml"


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

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.fastai"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)

    # Save an Learner
    fastai_learner.save(model_data_path)

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
    mlflow_model.add_flavor(FLAVOR_NAME, xgb_version=fastai.__version__, data=model_data_subpath)
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
              lgb_model=fastai_learner, conda_env=conda_env, **kwargs)


def _load_model(path):
    from fastai.basic_train import Learner
    model = Learner()
    return model.load(os.path.abspath(path))


class _FastaiModelWrapper:
    def __init__(self, learner):
        self.learner = learner

    def predict(self, dataframe):
        from fastai.basic_train import Learner
        return self.xgb_model.predict(xgb.DMatrix(dataframe))


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
    lgb_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.fastai"))
    return _load_model(path=lgb_model_file_path)


@experimental
def autolog():
    """
    Enable automatic logging from Keras to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    EarlyStopping Integration with Keras Automatic Logging

    MLflow will detect if an ``EarlyStopping`` callback is used in a ``fit()``/``fit_generator()``
    call, and if the ``restore_best_weights`` parameter is set to be ``True``, then MLflow will
    log the metrics associated with the restored model as a final, extra step. The epoch of the
    restored model will also be logged as the metric ``restored_epoch``.
    This allows for easy comparison between the actual metrics of the restored model and
    the metrics of other models.

    If ``restore_best_weights`` is set to be ``False``,
    then MLflow will not log an additional step.

    Regardless of ``restore_best_weights``, MLflow will also log ``stopped_epoch``,
    which indicates the epoch at which training stopped due to early stopping.

    If training does not end due to early stopping, then ``stopped_epoch`` will be logged as ``0``.

    MLflow will also log the parameters of the EarlyStopping callback,
    excluding ``mode`` and ``verbose``.
    """
    import keras

    class __MLflowKerasCallback(keras.callbacks.Callback):
        """
        Callback for auto-logging metrics and parameters.
        Records available logs after each epoch.
        Records model structural information as params when training begins
        """
        def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
            try_mlflow_log(mlflow.log_param, 'num_layers', len(self.model.layers))
            try_mlflow_log(mlflow.log_param, 'optimizer_name', type(self.model.optimizer).__name__)
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr if \
                    type(self.model.optimizer.lr) is float \
                    else keras.backend.eval(self.model.optimizer.lr)
                try_mlflow_log(mlflow.log_param, 'learning_rate', lr)
            if hasattr(self.model.optimizer, 'epsilon'):
                epsilon = self.model.optimizer.epsilon if \
                    type(self.model.optimizer.epsilon) is float \
                    else keras.backend.eval(self.model.optimizer.epsilon)
                try_mlflow_log(mlflow.log_param, 'epsilon', epsilon)

            sum_list = []
            self.model.summary(print_fn=sum_list.append)
            summary = '\n'.join(sum_list)
            try_mlflow_log(mlflow.set_tag, 'model_summary', summary)

            tempdir = tempfile.mkdtemp()
            try:
                summary_file = os.path.join(tempdir, "model_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(summary)
                try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
            finally:
                shutil.rmtree(tempdir)

        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            try_mlflow_log(mlflow.log_metrics, logs, step=epoch)

        def on_train_end(self, logs=None):
            try_mlflow_log(log_model, self.model, artifact_path='model')

    def _early_stop_check(callbacks):
        if LooseVersion(keras.__version__) < LooseVersion('2.3.0'):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {'monitor': callback.monitor,
                                        'min_delta': callback.min_delta,
                                        'patience': callback.patience,
                                        'baseline': callback.baseline,
                                        'restore_best_weights': callback.restore_best_weights}
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history):
        if callback:
            callback_attrs = _get_early_stop_callback_attrs(callback)
            if callback_attrs is None:
                return
            stopped_epoch, restore_best_weights, patience = callback_attrs
            try_mlflow_log(mlflow.log_metric, 'stopped_epoch', stopped_epoch)
            # Weights are restored only if early stopping occurs
            if stopped_epoch != 0 and restore_best_weights:
                restored_epoch = stopped_epoch - max(1, patience)
                try_mlflow_log(mlflow.log_metric, 'restored_epoch', restored_epoch)
                restored_metrics = {key: history.history[key][restored_epoch]
                                    for key in history.history.keys()}
                # Checking that a metric history exists
                metric_key = next(iter(history.history), None)
                if metric_key is not None:
                    last_epoch = len(history.history[metric_key])
                    try_mlflow_log(mlflow.log_metrics, restored_metrics, step=last_epoch)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        early_stop_callback = None

        # Checking if the 'callback' argument of the function is set
        if len(args) > callback_arg_index:
            tmp_list = list(args)
            early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
            tmp_list[callback_arg_index] += [__MLflowKerasCallback()]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs:
            early_stop_callback = _early_stop_check(kwargs['callbacks'])
            kwargs['callbacks'] += [__MLflowKerasCallback()]
        else:
            kwargs['callbacks'] = [__MLflowKerasCallback()]

        _log_early_stop_callback_params(early_stop_callback)

        history = original(self, *args, **kwargs)

        _log_early_stop_callback_metrics(early_stop_callback, history)

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)

        return history

    @gorilla.patch(keras.Model)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit')
        unlogged_params = ['self', 'x', 'y', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    @gorilla.patch(keras.Model)
    def fit_generator(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit_generator')
        unlogged_params = ['self', 'generator', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(keras.Model, 'fit', fit, settings=settings))
    gorilla.apply(gorilla.Patch(keras.Model, 'fit_generator', fit_generator, settings=settings))
