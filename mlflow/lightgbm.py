"""
The ``mlflow.lightgbm`` module provides an API for logging and loading LightGBM models.
This module exports LightGBM models with the following flavors:

LightGBM (native) format
    This is the main flavor that can be loaded back into LightGBM.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _lightgbm.Booster:
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster
.. _lightgbm.Booster.save_model:
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html
    #lightgbm.Booster.save_model
.. _lightgbm.train:
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm-train
.. _scikit-learn API:
    https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
"""
import os
import yaml
import json
import tempfile
import shutil
import inspect
import logging
import gorilla

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params


FLAVOR_NAME = "lightgbm"

_logger = logging.getLogger(__name__)


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import lightgbm as lgb

    return _mlflow_conda_env(
        additional_conda_deps=None,
        # LightGBM is not yet available via the default conda channels, so we install it via pip
        additional_pip_deps=[
            "lightgbm=={}".format(lgb.__version__),
        ],
        additional_conda_channels=None)


def save_model(lgb_model, path, conda_env=None, mlflow_model=None,
               signature: ModelSignature = None, input_example: ModelInputExample = None):
    """
    Save a LightGBM model to a path on the local file system.

    :param lgb_model: LightGBM model (an instance of `lightgbm.Booster`_) to be saved.
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
                                    'lightgbm==2.3.0'
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
    import lightgbm as lgb

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.lgb"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save a LightGBM model
    lgb_model.save_model(model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.lightgbm",
                        data=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, lgb_version=lgb.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(lgb_model, artifact_path, conda_env=None, registered_model_name=None,
              signature: ModelSignature=None, input_example: ModelInputExample=None,
              **kwargs):
    """
    Log a LightGBM model as an MLflow artifact for the current run.

    :param lgb_model: LightGBM model (an instance of `lightgbm.Booster`_) to be saved.
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
                                    'lightgbm==2.3.0'
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

    :param kwargs: kwargs to pass to `lightgbm.Booster.save_model`_ method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.lightgbm,
              registered_model_name=registered_model_name,
              lgb_model=lgb_model, conda_env=conda_env,
              signature=signature, input_example=input_example,
              **kwargs)


def _load_model(path):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``lightgbm`` flavor.
    """
    return _LGBModelWrapper(_load_model(path))


def load_model(model_uri):
    """
    Load a LightGBM model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A LightGBM model (an instance of `lightgbm.Booster`_).
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    lgb_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.lgb"))
    return _load_model(path=lgb_model_file_path)


class _LGBModelWrapper:
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model

    def predict(self, dataframe):
        return self.lgb_model.predict(dataframe)


@experimental
def autolog():
    """
    Enables automatic logging from LightGBM to MLflow. Logs the following.

    - parameters specified in `lightgbm.train`_.
    - metrics on each iteration (if ``valid_sets`` specified).
    - metrics at the best iteration (if ``early_stopping_rounds`` specified).
    - feature importance (both "split" and "gain") as JSON files and plots.
    - trained model.

    Note that the `scikit-learn API`_ is not supported.
    """
    import lightgbm
    import numpy as np

    @gorilla.patch(lightgbm)
    def train(*args, **kwargs):

        def record_eval_results(eval_results):
            """
            Create a callback function that records evaluation results.
            """
            def callback(env):
                res = {}
                for data_name, eval_name, value, _ in env.evaluation_result_list:
                    key = data_name + '-' + eval_name
                    res[key] = value

                eval_results.append(res)
            return callback

        def log_feature_importance_plot(features, importance, importance_type):
            """
            Log feature importance plot.
            """
            import matplotlib.pyplot as plt

            indices = np.argsort(importance)
            features = np.array(features)[indices]
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

        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        original = gorilla.get_original_attribute(lightgbm, 'train')

        # logging booster params separately via mlflow.log_params to extract key/value pairs
        # and make it easier to compare them across runs.
        params = args[0] if len(args) > 0 else kwargs['params']
        try_mlflow_log(mlflow.log_params, params)

        unlogged_params = ['params', 'train_set', 'valid_sets', 'valid_names', 'fobj', 'feval',
                           'init_model', 'evals_result', 'learning_rates', 'callbacks']

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
            try_mlflow_log(mlflow.log_metric, 'stopped_iteration', len(eval_results))
            # best_iteration is set even if training does not stop early.
            try_mlflow_log(mlflow.log_metric, 'best_iteration', model.best_iteration)
            # iteration starts from 1 in LightGBM.
            try_mlflow_log(mlflow.log_metrics, eval_results[model.best_iteration - 1],
                           step=extra_step)

        # logging feature importance as artifacts.
        for imp_type in ['split', 'gain']:
            features = model.feature_name()
            importance = model.feature_importance(importance_type=imp_type)
            try:
                log_feature_importance_plot(features, importance, imp_type)
            except Exception:  # pylint: disable=broad-except
                _logger.exception('Failed to log feature importance plot. LightGBM autologging '
                                  'will ignore the failure and continue. Exception: ')

            imp = {ft: imp for ft, imp in zip(features, importance.tolist())}
            tmpdir = tempfile.mkdtemp()
            try:
                filepath = os.path.join(tmpdir, 'feature_importance_{}.json'.format(imp_type))
                with open(filepath, 'w') as f:
                    json.dump(imp, f, indent=2)
                try_mlflow_log(mlflow.log_artifact, filepath)
            finally:
                shutil.rmtree(tmpdir)

        try_mlflow_log(log_model, model, artifact_path='model')

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)
        return model

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(lightgbm, 'train', train, settings=settings))
