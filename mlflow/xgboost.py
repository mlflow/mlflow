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
from packaging.version import Version
import os
import shutil
import json
import yaml
import tempfile
import inspect
import logging
from copy import deepcopy

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    exception_safe_function,
    get_mlflow_run_params_for_fn_args,
    INPUT_EXAMPLE_SAMPLE_ROWS,
    resolve_input_example_and_signature,
    InputExampleInfo,
    ENSURE_AUTOLOGGING_ENABLED_TEXT,
    batch_metrics_logger,
    MlflowAutologgingQueueingClient,
)

# Pylint doesn't detect objects used in class keyword arguments (e.g., metaclass) and considers
# `ExceptionSafeAbstractClass` as 'unused-import': https://github.com/PyCQA/pylint/issues/1630
# To avoid this bug, disable 'unused-import' on this line.
from mlflow.utils.autologging_utils import (  # pylint: disable=unused-import
    ExceptionSafeAbstractClass,
)

from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "xgboost"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("xgboost")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    xgb_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save an XGBoost model to a path on the local file system.

    :param xgb_model: XGBoost model (an instance of `xgboost.Booster`_) to be saved.
                      Note that models that implement the `scikit-learn API`_  are not supported.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    import xgboost as xgb

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

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

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.xgboost",
        data=model_data_subpath,
        env=_CONDA_ENV_FILE_NAME,
    )
    mlflow_model.add_flavor(FLAVOR_NAME, xgb_version=xgb.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    xgb_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log an XGBoost model as an MLflow artifact for the current run.

    :param xgb_model: XGBoost model (an instance of `xgboost.Booster`_) to be saved.
                      Note that models that implement the `scikit-learn API`_  are not supported.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to `xgboost.Booster.save_model`_ method.
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.xgboost,
        registered_model_name=registered_model_name,
        xgb_model=xgb_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


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
@autologging_integration(FLAVOR_NAME)
def autolog(
    importance_types=None,
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=W0102,unused-argument
    """
    Enables (or disables) and configures autologging from XGBoost to MLflow. Logs the following:

    - parameters specified in `xgboost.train`_.
    - metrics on each iteration (if ``evals`` specified).
    - metrics at the best iteration (if ``early_stopping_rounds`` specified).
    - feature importance as JSON files and plots.
    - trained model, including:
        - an example of valid input.
        - inferred signature of the inputs and outputs of the model.

    Note that the `scikit-learn API`_ is not supported.

    :param importance_types: Importance types to log. If unspecified, defaults to ``["weight"]``.
    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with XGBoost model artifacts during training. If
                               ``False``, input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with XGBoost model artifacts during training. If ``False``,
                                 signatures are not logged.
                                 Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param disable: If ``True``, disables the XGBoost autologging integration. If ``False``,
                    enables the XGBoost autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      xgboost that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during XGBoost
                   autologging. If ``False``, show all events and warnings during XGBoost
                   autologging.
    """
    import xgboost
    import numpy as np

    if importance_types is None:
        importance_types = ["weight"]

    # Patching this function so we can get a copy of the data given to DMatrix.__init__
    #   to use as an input example and for inferring the model signature.
    #   (there is no way to get the data back from a DMatrix object)
    # We store it on the DMatrix object so the train function is able to read it.
    def __init__(original, self, *args, **kwargs):
        data = args[0] if len(args) > 0 else kwargs.get("data")

        if data is not None:
            try:
                if isinstance(data, str):
                    raise Exception(
                        "cannot gather example input when dataset is loaded from a file."
                    )

                input_example_info = InputExampleInfo(
                    input_example=deepcopy(data[:INPUT_EXAMPLE_SAMPLE_ROWS])
                )
            except Exception as e:
                input_example_info = InputExampleInfo(error_msg=str(e))

            setattr(self, "input_example_info", input_example_info)

        original(self, *args, **kwargs)

    def train(original, *args, **kwargs):
        def record_eval_results(eval_results, metrics_logger):
            """
            Create a callback function that records evaluation results.
            """
            # TODO: Remove `replace("SNAPSHOT", "dev")` once the following issue is addressed:
            #       https://github.com/dmlc/xgboost/issues/6984
            if Version(xgboost.__version__.replace("SNAPSHOT", "dev")) >= Version("1.3.0"):
                # In xgboost >= 1.3.0, user-defined callbacks should inherit
                # `xgboost.callback.TrainingCallback`:
                # https://xgboost.readthedocs.io/en/latest/python/callbacks.html#defining-your-own-callback  # noqa

                class Callback(
                    xgboost.callback.TrainingCallback, metaclass=ExceptionSafeAbstractClass,
                ):
                    def after_iteration(self, model, epoch, evals_log):
                        """
                        Run after each iteration. Return True when training should stop.
                        """
                        # `evals_log` is a nested dict (type: Dict[str, Dict[str, List[float]]])
                        # that looks like this:
                        # {
                        #   "train": {
                        #     "auc": [0.5, 0.6, 0.7, ...],
                        #     ...
                        #   },
                        #   ...
                        # }
                        evaluation_result_dict = {}
                        for data_name, metric_dict in evals_log.items():
                            for metric_name, metric_values_on_each_iter in metric_dict.items():
                                key = "{}-{}".format(data_name, metric_name)
                                # The last element in `metric_values_on_each_iter` corresponds to
                                # the meric on the current iteration
                                evaluation_result_dict[key] = metric_values_on_each_iter[-1]

                        metrics_logger.record_metrics(evaluation_result_dict, epoch)
                        eval_results.append(evaluation_result_dict)

                        # Return `False` to indicate training should not stop
                        return False

                return Callback()

            else:

                @exception_safe_function
                def callback(env):
                    metrics_logger.record_metrics(dict(env.evaluation_result_list), env.iteration)
                    eval_results.append(dict(env.evaluation_result_list))

                return callback

        def log_feature_importance_plot(features, importance, importance_type):
            """
            Log feature importance plot.
            """
            import matplotlib.pyplot as plt
            from cycler import cycler

            features = np.array(features)

            # Structure the supplied `importance` values as a `num_features`-by-`num_classes` matrix
            importances_per_class_by_feature = np.array(importance)
            if importances_per_class_by_feature.ndim <= 1:
                # In this case, the supplied `importance` values are not given per class. Rather,
                # one importance value is given per feature. For consistency with the assumed
                # `num_features`-by-`num_classes` matrix structure, we coerce the importance
                # values to a `num_features`-by-1 matrix
                indices = np.argsort(importance)
                # Sort features and importance values by magnitude during transformation to a
                # `num_features`-by-`num_classes` matrix
                features = features[indices]
                importances_per_class_by_feature = np.array(
                    [[importance] for importance in importances_per_class_by_feature[indices]]
                )
                # In this case, do not include class labels on the feature importance plot because
                # only one importance value has been provided per feature, rather than an
                # one importance value for each class per feature
                label_classes_on_plot = False
            else:
                importance_value_magnitudes = np.abs(importances_per_class_by_feature).sum(axis=1)
                indices = np.argsort(importance_value_magnitudes)
                features = features[indices]
                importances_per_class_by_feature = importances_per_class_by_feature[indices]
                label_classes_on_plot = True

            num_classes = importances_per_class_by_feature.shape[1]
            num_features = len(features)

            # If num_features > 10, increase the figure height to prevent the plot
            # from being too dense.
            w, h = [6.4, 4.8]  # matplotlib's default figure size
            h = h + 0.1 * num_features if num_features > 10 else h
            h = h + 0.1 * num_classes if num_classes > 1 else h
            fig, ax = plt.subplots(figsize=(w, h))
            # When importance values are provided for each class per feature, we want to ensure
            # that the same color is used for all bars in the bar chart that have the same class
            colors_to_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][:num_classes]
            color_cycler = cycler(color=colors_to_cycle)
            ax.set_prop_cycle(color_cycler)

            # The following logic operates on one feature at a time, adding a bar to the bar chart
            # for each class that reflects the importance of the feature to predictions of that
            # class
            feature_ylocs = np.arange(num_features)
            # Define offsets on the y-axis that are used to evenly space the bars for each class
            # around the y-axis position of each feature
            offsets_per_yloc = np.linspace(-0.5, 0.5, num_classes) / 2 if num_classes > 1 else [0]
            for feature_idx, (feature_yloc, importances_per_class) in enumerate(
                zip(feature_ylocs, importances_per_class_by_feature)
            ):
                for class_idx, (offset, class_importance) in enumerate(
                    zip(offsets_per_yloc, importances_per_class)
                ):
                    (bar,) = ax.barh(
                        feature_yloc + offset,
                        class_importance,
                        align="center",
                        # Set the bar height such that importance value bars for a particular
                        # feature are spaced properly relative to each other (no overlap or gaps)
                        # and relative to importance value bars for other features
                        height=(0.5 / max(num_classes - 1, 1)),
                    )
                    if label_classes_on_plot and feature_idx == 0:
                        # Only set a label the first time a bar for a particular class is plotted to
                        # avoid duplicate legend entries. If we were to set a label for every bar,
                        # the legend would contain `num_features` labels for each class.
                        bar.set_label("Class {}".format(class_idx))

            ax.set_yticks(feature_ylocs)
            ax.set_yticklabels(features)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance ({})".format(importance_type))
            if label_classes_on_plot:
                ax.legend()
            fig.tight_layout()

            tmpdir = tempfile.mkdtemp()
            try:
                # pylint: disable=undefined-loop-variable
                filepath = os.path.join(tmpdir, "feature_importance_{}.png".format(imp_type))
                fig.savefig(filepath)
                mlflow.log_artifact(filepath)
            finally:
                plt.close(fig)
                shutil.rmtree(tmpdir)

        autologging_client = MlflowAutologgingQueueingClient()
        # logging booster params separately to extract key/value pairs and make it easier to
        # compare them across runs.
        booster_params = args[0] if len(args) > 0 else kwargs["params"]
        autologging_client.log_params(run_id=mlflow.active_run().info.run_id, params=booster_params)

        unlogged_params = [
            "params",
            "dtrain",
            "evals",
            "obj",
            "feval",
            "evals_result",
            "xgb_model",
            "callbacks",
            "learning_rates",
        ]
        params_to_log_for_fn = get_mlflow_run_params_for_fn_args(
            original, args, kwargs, unlogged_params
        )
        autologging_client.log_params(
            run_id=mlflow.active_run().info.run_id, params=params_to_log_for_fn
        )

        param_logging_operations = autologging_client.flush(synchronous=False)

        all_arg_names = inspect.getargspec(original)[0]  # pylint: disable=W1505
        num_pos_args = len(args)

        # adding a callback that records evaluation results.
        eval_results = []
        callbacks_index = all_arg_names.index("callbacks")

        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            callback = record_eval_results(eval_results, metrics_logger)
            if num_pos_args >= callbacks_index + 1:
                tmp_list = list(args)
                tmp_list[callbacks_index] += [callback]
                args = tuple(tmp_list)
            elif "callbacks" in kwargs and kwargs["callbacks"] is not None:
                kwargs["callbacks"] += [callback]
            else:
                kwargs["callbacks"] = [callback]

            # training model
            model = original(*args, **kwargs)

            # If early_stopping_rounds is present, logging metrics at the best iteration
            # as extra metrics with the max step + 1.
            early_stopping_index = all_arg_names.index("early_stopping_rounds")
            early_stopping = (
                num_pos_args >= early_stopping_index + 1 or "early_stopping_rounds" in kwargs
            )
            if early_stopping:
                extra_step = len(eval_results)
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics={
                        "stopped_iteration": extra_step - 1,
                        "best_iteration": model.best_iteration,
                    },
                )
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics=eval_results[model.best_iteration],
                    step=extra_step,
                )
                early_stopping_logging_operations = autologging_client.flush(synchronous=False)

        # logging feature importance as artifacts.
        for imp_type in importance_types:
            imp = None
            try:
                imp = model.get_score(importance_type=imp_type)
                features, importance = zip(*imp.items())
                log_feature_importance_plot(features, importance, imp_type)
            except Exception:
                _logger.exception(
                    "Failed to log feature importance plot. XGBoost autologging "
                    "will ignore the failure and continue. Exception: "
                )

            if imp is not None:
                tmpdir = tempfile.mkdtemp()
                try:
                    filepath = os.path.join(tmpdir, "feature_importance_{}.json".format(imp_type))
                    with open(filepath, "w") as f:
                        json.dump(imp, f)
                    mlflow.log_artifact(filepath)
                finally:
                    shutil.rmtree(tmpdir)

        # dtrain must exist as the original train function already ran successfully
        dtrain = args[1] if len(args) > 1 else kwargs.get("dtrain")

        # it is possible that the dataset was constructed before the patched
        #   constructor was applied, so we cannot assume the input_example_info exists
        input_example_info = getattr(dtrain, "input_example_info", None)

        def get_input_example():
            if input_example_info is None:
                raise Exception(ENSURE_AUTOLOGGING_ENABLED_TEXT)
            if input_example_info.error_msg is not None:
                raise Exception(input_example_info.error_msg)
            return input_example_info.input_example

        def infer_model_signature(input_example):
            model_output = model.predict(xgboost.DMatrix(input_example))
            model_signature = infer_signature(input_example, model_output)
            return model_signature

        # Only log the model if the autolog() param log_models is set to True.
        if log_models:
            # Will only resolve `input_example` and `signature` if `log_models` is `True`.
            input_example, signature = resolve_input_example_and_signature(
                get_input_example,
                infer_model_signature,
                log_input_examples,
                log_model_signatures,
                _logger,
            )

            log_model(
                model, artifact_path="model", signature=signature, input_example=input_example,
            )

        param_logging_operations.await_completion()
        if early_stopping:
            early_stopping_logging_operations.await_completion()

        return model

    safe_patch(FLAVOR_NAME, xgboost, "train", train, manage_run=True)
    safe_patch(FLAVOR_NAME, xgboost.DMatrix, "__init__", __init__)
