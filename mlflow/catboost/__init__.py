"""
The ``mlflow.catboost`` module provides an API for logging and loading CatBoost models.
This module exports CatBoost models with the following flavors:

CatBoost (native) format
    This is the main flavor that can be loaded back into CatBoost.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _CatBoost:
    https://catboost.ai/docs/concepts/python-reference_catboost.html
.. _CatBoost.save_model:
    https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html
.. _CatBoostClassifier:
    https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
.. _CatBoostRanker:
    https://catboost.ai/docs/concepts/python-reference_catboostranker.html
.. _CatBoostRegressor:
    https://catboost.ai/docs/concepts/python-reference_catboostregressor.html
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import NumpyDataset, from_numpy
from mlflow.data.pandas_dataset import PandasDataset, from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    MlflowAutologgingQueueingClient,
    autologging_integration,
    batch_metrics_logger,
    get_autologging_config,
    resolve_input_example_and_signature,
    safe_patch,
)
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
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

if TYPE_CHECKING:
    import catboost as cb
    import numpy as np
    import pandas as pd
    import scipy as sp

    CATBOOST_X_DATA_TYPE = (
        list | cb.Pool | pd.DataFrame | pd.Series | np.ndarray | sp.sparse.spmatrix
    )

    CATBOOST_Y_DATA_TYPE = list | pd.DataFrame | pd.Series | np.ndarray

    CATBOOST_EVAL_SET_TYPE = (
        cb.Pool
        | tuple[CATBOOST_X_DATA_TYPE, CATBOOST_Y_DATA_TYPE]
        | list[cb.Pool]
        | list[tuple[CATBOOST_X_DATA_TYPE, CATBOOST_Y_DATA_TYPE]]
    )

FLAVOR_NAME = "catboost"
_MODEL_TYPE_KEY = "model_type"
_SAVE_FORMAT_KEY = "save_format"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.cb"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("catboost")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    cb_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """Save a CatBoost model to a path on the local file system.

    Args:
        cb_model: CatBoost model (an instance of `CatBoost`_, `CatBoostClassifier`_,
            `CatBoostRanker`_, or `CatBoostRegressor`_) to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        kwargs: kwargs to pass to `CatBoost.save_model`_ method.

    """
    import catboost as cb

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _CatboostModelWrapper(cb_model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_path = os.path.join(path, _MODEL_BINARY_FILE_NAME)
    cb_model.save_model(model_data_path, **kwargs)

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.catboost",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )

    flavor_conf = {
        _MODEL_TYPE_KEY: cb_model.__class__.__name__,
        _SAVE_FORMAT_KEY: kwargs.get("format", "cbm"),
        **model_bin_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        catboost_version=cb.__version__,
        code=code_dir_subpath,
        **flavor_conf,
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
    cb_model,
    artifact_path: str | None = None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    **kwargs,
):
    """Log a CatBoost model as an MLflow artifact for the current run.

    Args:
        cb_model: CatBoost model (an instance of `CatBoost`_, `CatBoostClassifier`_,
            `CatBoostRanker`_, or `CatBoostRegressor`_) to be saved.
        artifact_path: Deprecated. Use `name` instead.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
        registered_model_name: If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: kwargs to pass to `CatBoost.save_model`_ method.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.catboost,
        registered_model_name=registered_model_name,
        cb_model=cb_model,
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


def _init_model(model_type):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    model_types = {
        c.__name__: c for c in [CatBoost, CatBoostClassifier, CatBoostRegressor]
    }

    with contextlib.suppress(ImportError):
        from catboost import CatBoostRanker

        model_types[CatBoostRanker.__name__] = CatBoostRanker

    if model_type not in model_types:
        raise TypeError(
            f"Invalid model type: '{model_type}'. Must be one of {list(model_types.keys())}"
        )

    return model_types[model_type]()


def _load_model(path, model_type, save_format):
    model = _init_model(model_type)
    model.load_model(os.path.abspath(path), save_format)
    return model


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``catboost`` flavor.
    """
    flavor_conf = _get_flavor_configuration(
        model_path=os.path.dirname(path), flavor_name=FLAVOR_NAME
    )
    return _CatboostModelWrapper(
        _load_model(
            path, flavor_conf.get(_MODEL_TYPE_KEY), flavor_conf.get(_SAVE_FORMAT_KEY)
        )
    )


def load_model(model_uri, dst_path=None):
    """Load a CatBoost model from a local file or a run.

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
        A CatBoost model (an instance of `CatBoost`_, `CatBoostClassifier`_, `CatBoostRanker`_,
        or `CatBoostRegressor`_)

    """
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    cb_model_file_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )
    return _load_model(
        cb_model_file_path,
        flavor_conf.get(_MODEL_TYPE_KEY),
        flavor_conf.get(_SAVE_FORMAT_KEY),
    )


class _CatboostModelWrapper:
    def __init__(self, cb_model):
        self.cb_model = cb_model

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.cb_model

    def predict(self, dataframe, params: dict[str, Any] | None = None):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        return self.cb_model.predict(dataframe)


# -------------------------------------------------------------------------------------
# MARK: Autologging
# -------------------------------------------------------------------------------------


# ---------------------------
# Dataset Logging Helpers
# ---------------------------


def _log_catboost_dataset(
    data, source, context, autologging_client, name=None
) -> PandasDataset | NumpyDataset | None:
    """
    Log a dataset to MLflow as a tracked input.

    Converts the data to an MLflow Dataset object and logs it with context tags
    (e.g., "train", "eval") to identify its role in training.
    """
    import numpy as np
    import pandas as pd

    # Wrap data in appropriate MLflow Dataset type based on input format
    if isinstance(data, pd.DataFrame):
        dataset = from_pandas(df=data, source=source, name=name)
    elif isinstance(data, np.ndarray):
        dataset = from_numpy(features=data, source=source, name=name)
    else:
        _logger.warning(
            "Unrecognized dataset type %s. Dataset logging skipped.", type(data)
        )
        return None

    # Tag dataset with its context (train/eval) and log to the active run
    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value=context)]
    dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
    autologging_client.log_inputs(
        run_id=mlflow.active_run().info.run_id, datasets=[dataset_input]
    )
    return dataset


# ---------------------------
# Data Conversion Helpers
# ---------------------------


def _pool_to_dataframe(pool: cb.Pool) -> pd.DataFrame:
    """Convert a CatBoost Pool to a DataFrame with feature and label columns."""
    import numpy as np
    import pandas as pd

    # Extract raw feature matrix and labels from the Pool object
    features = pool.get_features()
    label = pool.get_label()

    # Create DataFrame with generic column names (feature_0, feature_1, ...)
    num_features = features.shape[1]
    columns = [f"feature_{i}" for i in range(num_features)]
    data = pd.DataFrame(data=features, columns=columns)  # type: ignore[index]

    # Append label column if present (Pool may be unlabeled for inference)
    if label is not None:
        data["label"] = label
    return data


def _standardize_catboost_input(
    X: CATBOOST_X_DATA_TYPE, y: CATBOOST_Y_DATA_TYPE | None
) -> pd.DataFrame:
    """
    Standardize CatBoost input to a DataFrame with feature and label columns.

    CatBoost accepts many input formats (Pool, DataFrame, ndarray, list, sparse).
    This normalizes them all to a DataFrame for consistent dataset logging.
    """
    import catboost as cb
    import numpy as np
    import pandas as pd
    import scipy as sp

    # Pool objects have their own extraction method
    if isinstance(X, cb.Pool):
        return _pool_to_dataframe(pool=X)

    # Convert various array-like formats to DataFrame
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, (pd.Series, np.ndarray, list, sp.sparse.spmatrix)):
        # Infer number of features from first row, create generic column names
        df = pd.DataFrame(data=X, columns=[f"feature_{i}" for i in range(len(X[0]))])  # type: ignore[index]
    else:
        raise ValueError(f"Unsupported input type: {type(X)}")

    # Attach labels if provided (for training data logging)
    if y is not None and isinstance(y, (pd.Series, np.ndarray)):
        df["label"] = y

    return df


# ---------------------------
# Eval Set Logging
# ---------------------------


def _log_eval_set_list(eval_set, source, autologging_client) -> None:
    """
    Log a list of evaluation datasets (Pools or (X, y) tuples).

    CatBoost supports multiple eval sets for tracking validation metrics
    on different data splits. Each is logged with a unique context tag.
    """
    import catboost as cb

    for i, es in enumerate(eval_set):
        # Handle Pool objects - extract features/labels to DataFrame
        if isinstance(es, cb.Pool):
            _log_catboost_dataset(
                data=_pool_to_dataframe(es),
                source=source,
                context=f"eval_{i}",
                autologging_client=autologging_client,
            )
        # Handle (X, y) tuples - combine features and labels
        elif isinstance(es, tuple) and len(es) >= 2:
            _log_catboost_dataset(
                data=_standardize_catboost_input(X=es[0], y=es[1]),
                source=source,
                context=f"eval_{i}",
                autologging_client=autologging_client,
            )


def _log_datasets(
    X: CATBOOST_X_DATA_TYPE,
    y: CATBOOST_Y_DATA_TYPE | None,
    eval_set: CATBOOST_EVAL_SET_TYPE | None,
    autologging_client,
) -> None:
    """
    Log training and evaluation datasets to MLflow.

    This is the main entry point for dataset logging during autolog.
    It handles the training data (X, y) and any eval_set variants.
    """
    import catboost as cb

    # Create a source reference linking datasets to the calling code location
    context_tags = context_registry.resolve_tags()
    source = CodeDatasetSource(tags=context_tags)

    # Always log training data
    data = _standardize_catboost_input(X=X, y=y)
    _log_catboost_dataset(
        data=data, source=source, context="train", autologging_client=autologging_client
    )

    # Skip eval set logging if not provided
    if eval_set is None:
        return

    # Handle different eval_set formats: single Pool, or list of Pool/(X,y) tuples
    if isinstance(eval_set, cb.Pool):
        _log_catboost_dataset(
            data=_pool_to_dataframe(pool=eval_set),
            source=source,
            context="eval",
            autologging_client=autologging_client,
        )
    elif isinstance(eval_set, list):
        _log_eval_set_list(
            eval_set=eval_set, source=source, autologging_client=autologging_client
        )


# ---------------------------
# Model Logging
# ---------------------------


def _log_catboost_model(
    model: cb.CatBoostClassifier | cb.CatBoostRegressor | cb.CatBoostRanker,
    X: cb.Pool | pd.DataFrame | np.ndarray,
    model_id: str | None,
) -> None:
    """
    Log the trained CatBoost model with optional signature and input example.

    This function handles the model artifact logging step of autolog, including
    signature inference from input/output samples and optional model registration.
    """
    import catboost as cb
    import pandas as pd

    input_example = None
    signature = None

    def get_input_example() -> pd.DataFrame | pd.Series:
        """
        Get a sample of the input data for signature inference.

        Returns:
            A DataFrame or Series containing a sample of the input data.
        """
        # For Pool objects, extract features and remove labels (not model input)
        if isinstance(X, cb.Pool):
            data = _pool_to_dataframe(pool=X)
            if "label" in data.columns:
                data = data.drop(columns=["label"])
        else:
            data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(data=X)

        # Return a small sample to avoid storing large examples
        return deepcopy(data[:INPUT_EXAMPLE_SAMPLE_ROWS])

    def infer_model_signature(input_ex) -> ModelSignature:
        """
        Infer the model signature from the input example.

        Args:
            input_ex: The input example data.

        Returns:
            A ModelSignature object representing the model's input and output.
        """
        # Run prediction to capture output schema alongside input schema
        return infer_signature(
            model_input=input_ex, model_output=model.predict(input_ex)
        )

    # Conditionally build input example and signature based on autolog config
    input_example, signature = resolve_input_example_and_signature(
        get_input_example=get_input_example,
        infer_model_signature=infer_model_signature,
        log_input_example=get_autologging_config(
            FLAVOR_NAME, "log_input_examples", False
        ),
        log_model_signature=get_autologging_config(
            flavor_name=FLAVOR_NAME,
            config_key="log_model_signatures",
            default_value=True,
        ),
        logger=_logger,
    )

    # Check if user wants to auto-register the model to the Model Registry
    registered_model_name_val = get_autologging_config(
        flavor_name=FLAVOR_NAME, config_key="registered_model_name", default_value=None
    )

    # Log the model artifact with all collected metadata
    log_model(
        cb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name_val,
        model_id=model_id,
    )


# ---------------------------
# Public Autolog API
# ---------------------------


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
):
    """
    Enables (or disables) and configures autologging from CatBoost to MLflow. Logs the following:

        - parameters from the trained model via ``get_all_params()``.
        - metrics on each iteration (if ``eval_set`` specified).
        - trained model.
        - datasets used for training and evaluation.

    Args:
        log_input_examples: If ``True``, input examples from training datasets are collected and
            logged along with CatBoost model artifacts during training. If ``False``, input
            examples are not logged. Note: Input examples are MLflow model attributes and are
            only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along with CatBoost
            model artifacts during training. If ``False``, signatures are not logged.
            Note: Model signatures are MLflow model attributes and are only collected if
            ``log_models`` is also ``True``.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
        log_datasets: If ``True``, train and validation dataset information is logged to MLflow
            Tracking if applicable. If ``False``, dataset information is not logged.
        disable: If ``True``, disables the CatBoost autologging integration. If ``False``,
            enables the CatBoost autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            catboost that have not been tested against this version of the MLflow client
            or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during CatBoost
            autologging. If ``False``, show all events and warnings during CatBoost autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name. The registered model is
            created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    import catboost as cb

    def fit_impl(original, self, *args, **kwargs):
        """
        Patched fit() implementation that wraps the original CatBoost fit method.

        This function is called instead of the original fit() when autologging is
        enabled. It logs datasets, injects a metrics callback, runs training,
        then logs params and the trained model.
        """
        from mlflow.catboost._autolog import AutologCallback

        # Queue client batches log calls for efficiency
        autologging_client = MlflowAutologgingQueueingClient()
        run_id = mlflow.active_run().info.run_id

        # Extract training arguments from positional or keyword args
        # CatBoost fit() signature: fit(X, y=None, ..., eval_set=None, ...)
        X = args[0] if len(args) > 0 else kwargs.get("X")
        if X is None:
            raise ValueError("Training data X is required but not provided")

        y = args[1] if len(args) > 1 else kwargs.get("y")
        if y is None and not isinstance(X, cb.Pool):
            raise ValueError(
                "Training labels y must be provided when X is not a CatBoost Pool"
            )

        # eval_set is the 9th positional arg (index 8) in CatBoost fit()
        eval_set = args[8] if len(args) > 8 else kwargs.get("eval_set")

        # Step 1: Log datasets before training starts
        if log_datasets and X is not None:
            try:
                _log_datasets(
                    X=X,
                    y=y,
                    eval_set=eval_set,
                    autologging_client=autologging_client,
                )
                autologging_client.flush(synchronous=False)
            except Exception as e:
                _logger.warning("Failed to log dataset information. Reason: %s", e)

        # Step 2: Set up metrics logging and run training
        # Pre-initialize model artifact to get model_id for associating metrics
        model_id = None
        if log_models:
            model_id = _initialize_logged_model("model", flavor=FLAVOR_NAME).model_id

        # Inject our callback to capture per-iteration metrics during training
        with batch_metrics_logger(run_id, model_id=model_id) as metrics_logger:
            callback = AutologCallback(metrics_logger)
            kwargs.setdefault("callbacks", []).append(callback)
            model = original(self, *args, **kwargs)

        # Step 3: Log all resolved parameters after training completes
        # get_all_params() returns full config including CatBoost's internal defaults
        try:
            all_params = model.get_all_params()
            autologging_client.log_params(run_id=run_id, params=all_params)
            autologging_client.flush(synchronous=False)
        except Exception as e:
            _logger.warning("Failed to log parameters. Reason: %s", e)

        # Step 4: Log the trained model artifact with signature and input example
        if log_models:
            try:
                _log_catboost_model(model, X, model_id)
            except Exception as e:
                _logger.warning("Failed to log model. Reason: %s", e)

        return model

    # Patch fit() for both Classifier and Regressor model classes
    # safe_patch wraps the original method and manages MLflow run lifecycle
    for model_class in [cb.CatBoostClassifier, cb.CatBoostRegressor]:
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=model_class,
            function_name="fit",
            patch_function=functools.partial(fit_impl),
            manage_run=True,  # Auto-create run if none exists
            extra_tags=extra_tags,
        )
