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

import functools
import inspect
import json
import logging
import os
import tempfile
from copy import deepcopy
from functools import partial
from typing import Any, Optional

import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.sklearn import _SklearnTrainingSession
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
    ENSURE_AUTOLOGGING_ENABLED_TEXT,
    INPUT_EXAMPLE_SAMPLE_ROWS,
    InputExampleInfo,
    MlflowAutologgingQueueingClient,
    autologging_integration,
    batch_metrics_logger,
    get_autologging_config,
    get_mlflow_run_params_for_fn_args,
    picklable_exception_safe_function,
    resolve_input_example_and_signature,
    safe_patch,
)
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "xgboost"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor. Calls to
        :func:`save_model()` and :func:`log_model()` produce a pip environment that, at minimum,
        contains these requirements.
    """
    return [_get_pinned_requirement("xgboost")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    xgb_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_format="xgb",
    metadata=None,
):
    """Save an XGBoost model to a path on the local file system.

    Args:
        xgb_model: XGBoost model (an instance of `xgboost.Booster`_ or models that implement the
            `scikit-learn API`_) to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        model_format: File format in which the model is to be saved.
        metadata: {{ metadata }}
    """
    import xgboost as xgb

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _XGBModelWrapper(xgb_model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata
    model_data_subpath = f"model.{model_format}"
    model_data_path = os.path.join(path, model_data_subpath)

    # Save an XGBoost model
    xgb_model.save_model(model_data_path)
    xgb_model_class = _get_fully_qualified_class_name(xgb_model)
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.xgboost",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        xgb_version=xgb.__version__,
        data=model_data_subpath,
        model_class=xgb_model_class,
        model_format=model_format,
        code=code_dir_subpath,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
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

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    xgb_model,
    artifact_path: Optional[str] = None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_format="xgb",
    metadata=None,
    name: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, Any]] = None,
    model_type: Optional[str] = None,
    step: int = 0,
    model_id: Optional[str] = None,
    **kwargs,
):
    """Log an XGBoost model as an MLflow artifact for the current run.

    Args:
        xgb_model: XGBoost model (an instance of `xgboost.Booster`_ or models that implement the
            `scikit-learn API`_) to be saved.
        artifact_path: Deprecated. Use `name` instead.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        model_format: File format in which the model is to be saved.
        metadata: {{ metadata }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: kwargs to pass to `xgboost.Booster.save_model`_ method.

    Returns
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.xgboost,
        registered_model_name=registered_model_name,
        xgb_model=xgb_model,
        model_format=model_format,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


def _load_model(path):
    """Load Model Implementation.

    Args:
        path: Local filesystem path to the MLflow Model with the ``xgboost`` flavor
        (MLflow < 1.22.0) or the top-level MLflow Model directory (MLflow >= 1.22.0).
    """
    model_dir = os.path.dirname(path) if os.path.isfile(path) else path
    flavor_conf = _get_flavor_configuration(model_path=model_dir, flavor_name=FLAVOR_NAME)

    # XGBoost models saved in MLflow >=1.22.0 have `model_class`
    # in the XGBoost flavor configuration to specify its XGBoost model class.
    # When loading models, we first get the XGBoost model from
    # its flavor configuration and then create an instance based on its class.
    model_class = flavor_conf.get("model_class", "xgboost.core.Booster")
    xgb_model_path = os.path.join(model_dir, flavor_conf.get("data"))

    model = _get_class_from_string(model_class)()
    model.load_model(xgb_model_path)
    return model


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``xgboost`` flavor.
    """
    return _XGBModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """Load an XGBoost model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        An XGBoost model. An instance of either `xgboost.Booster`_ or XGBoost scikit-learn
        models, depending on the saved model class specification.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    return _load_model(path=local_model_path)


class _XGBModelWrapper:
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model

    def get_raw_model(self):
        """
        Returns the underlying XGBoost model.
        """
        return self.xgb_model

    def predict(
        self,
        dataframe,
        params: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        import xgboost as xgb

        predict_fn = _wrapped_xgboost_model_predict_fn(self.xgb_model)
        params = params or {}
        # filter is applied inside predict_fn wrapper for xgb.Booster
        if not isinstance(self.xgb_model, xgb.Booster):
            # Exclude unrecognized parameters as feature store team has
            # dependency on this behavior. They might pass additional parameters
            # that cannot be passed to the model.
            params = _exclude_unrecognized_kwargs(predict_fn, params)
        return predict_fn(dataframe, **params)


def _exclude_unrecognized_kwargs(predict_fn, kwargs):
    filtered_kwargs = {}
    allowed_params = inspect.signature(predict_fn).parameters
    # avoid excluding kwargs when predict function uses args or kwargs
    if any(p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD for p in allowed_params.values()):
        return kwargs
    invalid_params = set()
    for key, value in kwargs.items():
        if key in allowed_params:
            filtered_kwargs[key] = value
        else:
            invalid_params.add(key)
    if invalid_params:
        _logger.warning(
            f"Params {invalid_params} are not accepted by the xgboost model, "
            "ignoring them during predict."
        )
    return filtered_kwargs


def _wrapped_xgboost_model_predict_fn(model, validate_features=True):
    """
    Wraps the predict method of the raw model to accept a DataFrame as input.
    """
    import xgboost as xgb

    if isinstance(model, xgb.Booster):
        # we need to wrap the predict function to accept data in pandas format
        def wrapped_predict_fn(data, *args, **kwargs):
            filtered_kwargs = _exclude_unrecognized_kwargs(model.predict, kwargs)
            return model.predict(
                xgb.DMatrix(data), *args, validate_features=validate_features, **filtered_kwargs
            )

        return wrapped_predict_fn
    elif isinstance(model, xgb.XGBModel):
        return partial(model.predict, validate_features=validate_features)
    else:
        return model.predict


def _wrapped_xgboost_model_predict_proba_fn(model, validate_features=True):
    import xgboost as xgb

    predict_proba_fn = getattr(model, "predict_proba", None)
    if isinstance(model, xgb.XGBModel) and predict_proba_fn is not None:
        return partial(predict_proba_fn, validate_features=validate_features)
    return predict_proba_fn


@autologging_integration(FLAVOR_NAME)
def autolog(
    importance_types=None,
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    model_format="xgb",
    extra_tags=None,
):
    """
    Enables (or disables) and configures autologging from XGBoost to MLflow. Logs the following:

        - parameters specified in `xgboost.train`_.
        - metrics on each iteration (if ``evals`` specified).
        - metrics at the best iteration (if ``early_stopping_rounds`` specified).
        - feature importance as JSON files and plots.
        - trained model, including:
            - an example of valid input.
            - inferred signature of the inputs and outputs of the model.

    Note that the `scikit-learn API`_ is now supported.

    Args:
        importance_types: Importance types to log. If unspecified, defaults to ``["weight"]``.
        log_input_examples: If ``True``, input examples from training datasets are collected and
            logged along with XGBoost model artifacts during training. If
            ``False``, input examples are not logged. Note: Input examples are MLflow model
            attributes and are only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along
            with XGBoost model artifacts during training. If ``False``,
            signatures are not logged.
            Note: Model signatures are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
            Input examples and model signatures, which are attributes of MLflow models,
            are also omitted when ``log_models`` is ``False``.
        log_datasets: If ``True``, train and validation dataset information is logged to MLflow
            Tracking if applicable. If ``False``, dataset information is not logged.
        disable: If ``True``, disables the XGBoost autologging integration. If ``False``,
            enables the XGBoost autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            xgboost that have not been tested against this version of the MLflow client
            or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during XGBoost
            autologging. If ``False``, show all events and warnings during XGBoost
            autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        model_format: File format in which the model is to be saved.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    import numpy as np
    import xgboost

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

            self.input_example_info = input_example_info

        original(self, *args, **kwargs)

    def train_impl(_log_models, _log_datasets, original, *args, **kwargs):
        def record_eval_results(eval_results, metrics_logger):
            """
            Create a callback function that records evaluation results.
            """
            # TODO: Remove `replace("SNAPSHOT", "dev")` once the following issue is addressed:
            #       https://github.com/dmlc/xgboost/issues/6984
            from mlflow.xgboost._autolog import IS_TRAINING_CALLBACK_SUPPORTED

            if IS_TRAINING_CALLBACK_SUPPORTED:
                from mlflow.xgboost._autolog import AutologCallback

                # In xgboost >= 1.3.0, user-defined callbacks should inherit
                # `xgboost.callback.TrainingCallback`:
                # https://xgboost.readthedocs.io/en/latest/python/callbacks.html#defining-your-own-callback
                return AutologCallback(metrics_logger, eval_results)
            else:
                from mlflow.xgboost._autolog import autolog_callback

                return picklable_exception_safe_function(
                    functools.partial(
                        autolog_callback, metrics_logger=metrics_logger, eval_results=eval_results
                    )
                )

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
                        bar.set_label(f"Class {class_idx}")

            ax.set_yticks(feature_ylocs)
            ax.set_yticklabels(features)
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance ({importance_type})")
            if label_classes_on_plot:
                ax.legend()
            fig.tight_layout()

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    filepath = os.path.join(tmpdir, f"feature_importance_{imp_type}.png")
                    fig.savefig(filepath)
                    mlflow.log_artifact(filepath)
                finally:
                    plt.close(fig)

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

        all_arg_names = _get_arg_names(original)
        num_pos_args = len(args)

        # adding a callback that records evaluation results.
        eval_results = []
        callbacks_index = all_arg_names.index("callbacks")

        run_id = mlflow.active_run().info.run_id

        dtrain = args[1] if len(args) > 1 else kwargs.get("dtrain")

        # Whether to automatically log the training dataset as a dataset artifact.
        dataset = None
        if _log_datasets and dtrain is not None:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(context_tags)

                dataset = _log_xgboost_dataset(dtrain, source, "train", autologging_client)
                evals = kwargs.get("evals")
                if evals is not None:
                    for d, name in evals:
                        _log_xgboost_dataset(d, source, "eval", autologging_client, name)
                dataset_logging_operations = autologging_client.flush(synchronous=False)
                dataset_logging_operations.await_completion()
            except Exception as e:
                _logger.warning(
                    "Failed to log dataset information to MLflow Tracking. Reason: %s", e
                )

        model_id = None
        if _log_models:
            model_id = mlflow.initialize_logged_model("model").model_id
        with batch_metrics_logger(run_id, model_id=model_id) as metrics_logger:
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

            # dtrain must exist as the original train function already ran successfully
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
                return infer_signature(input_example, model_output)

            # Only log the model if the autolog() param log_models is set to True.
            if _log_models:
                # Will only resolve `input_example` and `signature` if `log_models` is `True`.
                input_example, signature = resolve_input_example_and_signature(
                    get_input_example,
                    infer_model_signature,
                    log_input_examples,
                    log_model_signatures,
                    _logger,
                )

                registered_model_name = get_autologging_config(
                    FLAVOR_NAME, "registered_model_name", None
                )
                log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    model_format=model_format,
                    params=params_to_log_for_fn,
                    model_id=model_id,
                )

            # If early_stopping_rounds is present, logging metrics at the best iteration
            # as extra metrics with the max step + 1.
            early_stopping_index = all_arg_names.index("early_stopping_rounds")
            early_stopping = num_pos_args >= early_stopping_index + 1 or kwargs.get(
                "early_stopping_rounds"
            )
            if early_stopping:
                extra_step = len(eval_results)
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics={
                        "stopped_iteration": extra_step - 1,
                        "best_iteration": model.best_iteration,
                    },
                    dataset=dataset,
                    model_id=model_id,
                )
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics=eval_results[model.best_iteration],
                    step=extra_step,
                    dataset=dataset,
                    model_id=model_id,
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
                with tempfile.TemporaryDirectory() as tmpdir:
                    filepath = os.path.join(tmpdir, f"feature_importance_{imp_type}.json")
                    with open(filepath, "w") as f:
                        json.dump(imp, f)
                    mlflow.log_artifact(filepath)

        param_logging_operations.await_completion()
        if early_stopping:
            early_stopping_logging_operations.await_completion()

        return model

    def train(_log_models, _log_datasets, original, *args, **kwargs):
        current_sklearn_session = _SklearnTrainingSession.get_current_session()
        if current_sklearn_session is None or current_sklearn_session.should_log():
            return train_impl(_log_models, _log_datasets, original, *args, **kwargs)
        else:
            return original(*args, **kwargs)

    safe_patch(
        FLAVOR_NAME,
        xgboost,
        "train",
        functools.partial(train, log_models, log_datasets),
        manage_run=True,
        extra_tags=extra_tags,
    )
    # The `train()` method logs XGBoost models as Booster objects. When using XGBoost
    # scikit-learn models, we want to save / log models as their model classes. So we turn
    # off the log_models functionality in the `train()` method patched to `xgboost.sklearn`.
    # Instead the model logging is handled in `fit_mlflow_sklearn()` in `mlflow.sklearn._autolog()`,
    # where models are logged as XGBoost scikit-learn models after the `fit()` method returns.
    safe_patch(
        FLAVOR_NAME,
        xgboost.sklearn,
        "train",
        functools.partial(train, False, log_datasets),
        manage_run=True,
        extra_tags=extra_tags,
    )
    safe_patch(FLAVOR_NAME, xgboost.DMatrix, "__init__", __init__)

    # enable xgboost scikit-learn estimators autologging
    import mlflow.sklearn

    mlflow.sklearn._autolog(
        flavor_name=FLAVOR_NAME,
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
        log_models=log_models,
        log_datasets=log_datasets,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        max_tuning_runs=None,
        log_post_training_metrics=True,
        extra_tags=extra_tags,
    )


def _log_xgboost_dataset(xgb_dataset, source, context, autologging_client, name=None):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from scipy.sparse import issparse

    # dmatrix has a get_data method added in 1.7. skip for earlier versions.
    if Version(xgb.__version__) >= Version("1.7.0"):
        data = xgb_dataset.get_data()
        if isinstance(xgb_dataset, pd.DataFrame):
            dataset = from_pandas(df=data, source=source, name=name)
        elif issparse(data):
            arr_data = data.toarray() if issparse(data) else data
            dataset = from_numpy(features=arr_data, source=source, name=name)
        elif isinstance(data, np.ndarray):
            dataset = from_numpy(features=data, source=source, name=name)
        else:
            _logger.warning("Unrecognized dataset type %s. Dataset logging skipped.", type(data))
            return

        tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value=context)]
        dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)

        autologging_client.log_inputs(
            run_id=mlflow.active_run().info.run_id, datasets=[dataset_input]
        )
        return dataset
    else:
        _logger.warning(
            "Unable to log dataset information to MLflow Tracking.XGBoost version must be >= 1.7.0"
        )
