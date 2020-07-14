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
.. _xgboost.train:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
.. _scikit-learn API:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
"""
import os
import shutil
import json
import yaml
import tempfile
import inspect
import logging
import gorilla

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params

FLAVOR_NAME = "xgboost"

_logger = logging.getLogger(__name__)


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


def save_model(xgb_model, path, conda_env=None, mlflow_model=None,
               signature: ModelSignature=None, input_example: ModelInputExample=None):
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

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    """
    import xgboost as xgb

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    model_data_subpath = "model.xgb"
    model_data_path = os.path.join(path, model_data_subpath)

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
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(xgb_model, artifact_path, conda_env=None, registered_model_name=None,
              signature: ModelSignature=None, input_example: ModelInputExample=None,
              **kwargs):
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
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    :param kwargs: kwargs to pass to `xgboost.Booster.save_model`_ method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.xgboost,
              registered_model_name=registered_model_name,
              xgb_model=xgb_model, conda_env=conda_env,
              signature=signature, input_example=input_example, **kwargs)


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
def autolog(importance_types=['weight']):  # pylint: disable=W0102
    """
    Enables automatic logging from XGBoost to MLflow. Logs the following.

    - parameters specified in `xgboost.train`_.
    - metrics on each iteration (if ``evals`` specified).
    - metrics at the best iteration (if ``early_stopping_rounds`` specified).
    - feature importance as JSON files and plots.
    - trained model.

    Note that the `scikit-learn API`_ is not supported.

    :param importance_types: importance types to log.

    """
    import xgboost
    import numpy as np

    @gorilla.patch(xgboost)
    def train(*args, **kwargs):

        def record_eval_results(eval_results):
            """
            Create a callback function that records evaluation results.
            """
            def callback(env):
                eval_results.append(dict(env.evaluation_result_list))
            return callback

        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        def log_feature_importance_plot(features, importance, importance_type):
            """
            Log feature importance plot.
            """
            import matplotlib.pyplot as plt

            features = np.array(features)
            importance = np.array(importance)
            indices = np.argsort(importance)
            features = features[indices]
            importance = importance[indices]
            num_features = len(features)

            # If num_features > 10, increase the figure height to prevent the plot
            # from being too dense.
            w, h = [6.4, 4.8]  # matplotlib's default figure size
            h = h + 0.1 * num_features if num_features > 10 else h
            fig, ax = plt.subplots(figsize=(w, h))

            yloc = np.arange(num_features)
            ax.barh(yloc, importance, align='center', height=0.5)
            ax.set_yticks(yloc)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance ({})'.format(importance_type))
            fig.tight_layout()

            tmpdir = tempfile.mkdtemp()
            try:
                # pylint: disable=undefined-loop-variable
                filepath = os.path.join(tmpdir, 'feature_importance_{}.png'.format(imp_type))
                fig.savefig(filepath)
                try_mlflow_log(mlflow.log_artifact, filepath)
            finally:
                plt.close(fig)
                shutil.rmtree(tmpdir)

        original = gorilla.get_original_attribute(xgboost, 'train')

        # logging booster params separately via mlflow.log_params to extract key/value pairs
        # and make it easier to compare them across runs.
        params = args[0] if len(args) > 0 else kwargs['params']
        try_mlflow_log(mlflow.log_params, params)

        unlogged_params = ['params', 'dtrain', 'evals', 'obj', 'feval', 'evals_result',
                           'xgb_model', 'callbacks', 'learning_rates']
        log_fn_args_as_params(original, args, kwargs, unlogged_params)

        all_arg_names = inspect.getargspec(original)[0]  # pylint: disable=W1505
        num_pos_args = len(args)

        # adding a callback that records evaluation results.
        eval_results = []
        callbacks_index = all_arg_names.index('callbacks')
        callback = record_eval_results(eval_results)
        if num_pos_args >= callbacks_index + 1:
            tmp_list = list(args)
            tmp_list[callbacks_index] += [callback]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs and kwargs['callbacks'] is not None:
            kwargs['callbacks'] += [callback]
        else:
            kwargs['callbacks'] = [callback]

        # training model
        model = original(*args, **kwargs)

        # logging metrics on each iteration.
        for idx, metrics in enumerate(eval_results):
            try_mlflow_log(mlflow.log_metrics, metrics, step=idx)

        # If early_stopping_rounds is present, logging metrics at the best iteration
        # as extra metrics with the max step + 1.
        early_stopping_index = all_arg_names.index('early_stopping_rounds')
        early_stopping = (num_pos_args >= early_stopping_index + 1 or
                          'early_stopping_rounds' in kwargs)
        if early_stopping:
            extra_step = len(eval_results)
            try_mlflow_log(mlflow.log_metric, 'stopped_iteration', len(eval_results) - 1)
            try_mlflow_log(mlflow.log_metric, 'best_iteration', model.best_iteration)
            try_mlflow_log(mlflow.log_metrics, eval_results[model.best_iteration],
                           step=extra_step)

        # logging feature importance as artifacts.
        for imp_type in importance_types:
            imp = model.get_score(importance_type=imp_type)
            features, importance = zip(*imp.items())
            try:
                log_feature_importance_plot(features, importance, imp_type)
            except Exception:  # pylint: disable=broad-except
                _logger.exception('Failed to log feature importance plot. LightGBM autologging '
                                  'will ignore the failure and continue. Exception: ')

            tmpdir = tempfile.mkdtemp()
            try:
                filepath = os.path.join(tmpdir, 'feature_importance_{}.json'.format(imp_type))
                with open(filepath, 'w') as f:
                    json.dump(imp, f)
                try_mlflow_log(mlflow.log_artifact, filepath)
            finally:
                shutil.rmtree(tmpdir)

        try_mlflow_log(log_model, model, artifact_path='model')

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)
        return model

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(xgboost, 'train', train, settings=settings))
