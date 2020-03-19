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
import gorilla
import tempfile
import shutil

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
    fastai_learner.export(model_data_path)

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
        # TODO
        from fastai.basic_train import Learner
        return dataframe


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


@experimental
def autolog():
    """
    Enable automatic logging from Fastai to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    EarlyStopping Integration with Fastai Automatic Logging

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
    from fastai.basic_train import LearnerCallback, Learner
    from fastai.callbacks.hooks import model_summary, layers_info
    from fastai.callbacks import EarlyStoppingCallback, OneCycleScheduler

    class __MLflowFastaiCallback(LearnerCallback):
        """
        Callback for auto-logging metrics and parameters.
        Records available logs after each epoch.
        Records model structural information as params when training begins
        """
        def __init__(self, learner):
            super().__init__(learner)
            self.learner = learner
            self.opt = self.learn.opt
            self.metrics_names = ['train_loss', 'valid_loss'] + [o.__name__ for o in learner.metrics]

        def on_epoch_end(self, epoch, **kwargs):
            """
            Log loss and other metrics values after each epoch
            """
            if kwargs['smooth_loss'] is None or kwargs["last_metrics"] is None:
                return
            metrics = [kwargs['smooth_loss']] + kwargs["last_metrics"]
            metrics = map(float, metrics)
            metrics = dict(zip(self.metrics_names, metrics))
            try_mlflow_log(mlflow.log_metrics, metrics, step=epoch)

        def on_train_begin(self):
            info = layers_info(self.learner)
            try_mlflow_log(mlflow.log_param, 'num_layers', len(info))
            try_mlflow_log(mlflow.log_param, 'optimizer_name', type(self.opt).__name__)

            # TODO: Log all opt params

            if hasattr(self.opt, 'lr'):
                try_mlflow_log(mlflow.log_param, 'learning_rate', self.opt.lr)

            if hasattr(self.opt, 'mom'):
                try_mlflow_log(mlflow.log_param, 'momentum', self.opt.mom)

            summary = model_summary(self.learner)
            try_mlflow_log(mlflow.set_tag, 'model_summary', summary)

            tempdir = tempfile.mkdtemp()
            try:
                summary_file = os.path.join(tempdir, "model_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(summary)
                try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
            finally:
                shutil.rmtree(tempdir)

        def on_train_end(self, logs=None):
            try_mlflow_log(log_model, self.model, artifact_path='model')

    def _find_callback_of_type(callback_type, callbacks):
        for callback in callbacks:
            if isinstance(callback, callback_type):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {'early_stop_monitor': callback.monitor,
                                        'early_stop_min_delta': callback.min_delta,
                                        'early_stop_patience': callback.patience,
                                        'early_stop_mode': callback.mode}
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _log_one_cycle_callback_params(callback):
        if callback:
            try:
                params = {
                    'one_cycle_l_max': callback.lr_max,
                    'one_cycle_div_factor': callback.div_factor,
                    'one_cycle_pct_start': callback.pct_start,
                    'one_cycle_final_div': callback.final_div
                }
                try_mlflow_log(mlflow.log_params, params)
            except Exception:  # pylint: disable=W0703
                return

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        log_fn_args_as_params(original, args, kwargs, unlogged_params)

        callbacks = [cb(self) for cb in self.callback_fns]

        # Checking if the 'callback' argument of the function is set
        if len(args) > callback_arg_index:
            tmp_list = list(args)
            callbacks += list(args[callback_arg_index])
            tmp_list[callback_arg_index] += [__MLflowFastaiCallback(self)]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs:
            callbacks += list(kwargs['callbacks'])
            kwargs['callbacks'] += [__MLflowFastaiCallback(self)]
        else:
            kwargs['callbacks'] = [__MLflowFastaiCallback(self)]

        early_stop_callback = _find_callback_of_type(EarlyStoppingCallback, callbacks)
        one_cycle_callback = _find_callback_of_type(OneCycleScheduler, callbacks)

        _log_one_cycle_callback_params(one_cycle_callback)
        _log_early_stop_callback_params(early_stop_callback)

        result = original(self, *args, **kwargs)

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)

        return result

    @gorilla.patch(Learner)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(Learner, 'fit')
        unlogged_params = ['self', 'callbacks']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 3)

    @gorilla.patch(Learner)
    def fit_one_cycle(self, *args, **kwargs):
        original = gorilla.get_original_attribute(Learner, 'fit_one_cycle')
        unlogged_params = ['self', 'callbacks']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 7)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(Learner, 'fit', fit, settings=settings))
    gorilla.apply(gorilla.Patch(Learner, 'fit_one_cycle', fit_one_cycle, settings=settings))
