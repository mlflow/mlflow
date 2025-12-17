"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    NOTE: The `mlflow.pyfunc` flavor is only added for scikit-learn models that define `predict()`,
    since `predict()` is required for pyfunc model inference.
"""

import functools
import inspect
import logging
import os
import pickle
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _inspect_original_var_name, gorilla
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    MlflowAutologgingQueueingClient,
    _get_new_training_session_class,
    autologging_integration,
    disable_autologging,
    get_autologging_config,
    get_instance_method_first_arg_value,
    resolve_input_example_and_signature,
    safe_patch,
    update_wrapper_extended,
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
from mlflow.utils.mlflow_tags import (
    MLFLOW_AUTOLOGGING,
    MLFLOW_DATASET_CONTEXT,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "sklearn"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [SERIALIZATION_FORMAT_PICKLE, SERIALIZATION_FORMAT_CLOUDPICKLE]

_logger = logging.getLogger(__name__)
_SklearnTrainingSession = _get_new_training_session_class()

_MODEL_DATA_SUBPATH = "model.pkl"


def _gen_estimators_to_patch():
    from mlflow.sklearn.utils import (
        _all_estimators,
        _get_meta_estimators_for_autologging,
    )

    _, estimators_to_patch = zip(*_all_estimators())
    # Ensure that relevant meta estimators (e.g. GridSearchCV, Pipeline) are selected
    # for patching if they are not already included in the output of `all_estimators()`
    estimators_to_patch = set(estimators_to_patch).union(
        set(_get_meta_estimators_for_autologging())
    )
    # Exclude certain preprocessing & feature manipulation estimators from patching. These
    # estimators represent data manipulation routines (e.g., normalization, label encoding)
    # rather than ML algorithms. Accordingly, we should not create MLflow runs and log
    # parameters / metrics for these routines, unless they are captured as part of an ML pipeline
    # (via `sklearn.pipeline.Pipeline`)
    excluded_module_names = [
        "sklearn.preprocessing",
        "sklearn.impute",
        "sklearn.feature_extraction",
        "sklearn.feature_selection",
    ]

    excluded_class_names = [
        "sklearn.compose._column_transformer.ColumnTransformer",
    ]

    return [
        estimator
        for estimator in estimators_to_patch
        if not any(
            estimator.__module__.startswith(excluded_module_name)
            or (estimator.__module__ + "." + estimator.__name__) in excluded_class_names
            for excluded_module_name in excluded_module_names
        )
    ]


def get_default_pip_requirements(include_cloudpickle=False):
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("scikit-learn", module="sklearn")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="scikit-learn"))
def save_model(
    sk_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
    metadata=None,
):
    """
    Save a scikit-learn model to a path on the local file system. Produces a MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    Args:
        sk_model: scikit-learn model to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        serialization_format: The format in which to serialize the model. This should be one of
            the formats listed in
            ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
            format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
            provides better cross-system compatibility by identifying and
            packaging code dependencies with the serialized model.

        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        pyfunc_predict_fn: The name of the prediction function to use for inference with the
            pyfunc representation of the resulting MLflow Model. Current supported functions
            are: ``"predict"``, ``"predict_proba"``, ``"predict_log_proba"``,
            ``"predict_joint_log_proba"``, and ``"score"``.
        metadata: {{ metadata }}

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree

        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)

        # Save the model in cloudpickle format
        # set path to location for persistence
        sk_path_dir_1 = ...
        mlflow.sklearn.save_model(
            sk_model,
            sk_path_dir_1,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # save the model in pickle format
        # set path to location for persistence
        sk_path_dir_2 = ...
        mlflow.sklearn.save_model(
            sk_model,
            sk_path_dir_2,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
    """
    import sklearn

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                f"Unrecognized serialization format: {serialization_format}. Please specify one"
                f" of the following supported formats: {SUPPORTED_SERIALIZATION_FORMATS}."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    _validate_and_prepare_target_save_path(path)
    code_path_subdir = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _SklearnModelWrapper(sk_model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_subpath = _MODEL_DATA_SUBPATH
    model_data_path = os.path.join(path, model_data_subpath)
    _save_model(
        sk_model=sk_model,
        output_path=model_data_path,
        serialization_format=serialization_format,
    )

    # `PyFuncModel` only works for sklearn models that define a predict function

    if hasattr(sk_model, pyfunc_predict_fn):
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.sklearn",
            model_path=model_data_subpath,
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_path_subdir,
            predict_fn=pyfunc_predict_fn,
        )
    else:
        _logger.warning(
            f"Model was missing function: {pyfunc_predict_fn}. Not logging python_function flavor!"
        )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sklearn_version=sklearn.__version__,
        serialization_format=serialization_format,
        code=code_path_subdir,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            include_cloudpickle = serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
            default_reqs = get_default_pip_requirements(include_cloudpickle)
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
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


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="scikit-learn"))
def log_model(
    sk_model,
    artifact_path: str | None = None,
    conda_env=None,
    code_paths=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
    metadata=None,
    # New arguments
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    name: str | None = None,
):
    """
    Log a scikit-learn model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    Args:
        sk_model: scikit-learn model to be saved.
        artifact_path: Deprecated. Use `name` instead.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        serialization_format: The format in which to serialize the model. This should be one of
            the formats listed in
            ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
            format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
            provides better cross-system compatibility by identifying and
            packaging code dependencies with the serialized model.
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
        pyfunc_predict_fn: The name of the prediction function to use for inference with the
            pyfunc representation of the resulting MLflow Model. Current supported functions
            are: ``"predict"``, ``"predict_proba"``, ``"predict_log_proba"``,
            ``"predict_joint_log_proba"``, and ``"score"``.
        metadata: {{ metadata }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        name: {{ name }}

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
        from sklearn.datasets import load_iris
        from sklearn import tree

        with mlflow.start_run():
            # load dataset and train model
            iris = load_iris()
            sk_model = tree.DecisionTreeClassifier()
            sk_model = sk_model.fit(iris.data, iris.target)

            # log model params
            mlflow.log_param("criterion", sk_model.criterion)
            mlflow.log_param("splitter", sk_model.splitter)
            signature = infer_signature(iris.data, sk_model.predict(iris.data))

            # log model
            mlflow.sklearn.log_model(sk_model, name="sk_models", signature=signature)

    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.sklearn,
        sk_model=sk_model,
        conda_env=conda_env,
        code_paths=code_paths,
        serialization_format=serialization_format,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        pyfunc_predict_fn=pyfunc_predict_fn,
        metadata=metadata,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
    )


def _load_model_from_local_file(path, serialization_format):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system.

    Args:
        path: Local filesystem path to the MLflow Model saved with the ``sklearn`` flavor
        serialization_format: The format in which the model was serialized. This should be one of
            the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
            ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    # TODO: we could validate the scikit-learn version here
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                f"Unrecognized serialization format: {serialization_format}. Please specify one"
                f" of the following supported formats: {SUPPORTED_SERIALIZATION_FORMATS}."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle cannot necessarily be deserialized using Pickle;
        # That's why we check the serialization format of the model before deserializing
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    if os.path.isfile(path):
        # Scikit-learn models saved in older versions of MLflow (<= 1.9.1) specify the ``data``
        # field within the pyfunc flavor configuration. For these older models, the ``path``
        # parameter of ``_load_pyfunc()`` refers directly to a serialized scikit-learn model
        # object. In this case, we assume that the serialization format is ``pickle``, since
        # the model loading procedure in older versions of MLflow used ``pickle.load()``.
        serialization_format = SERIALIZATION_FORMAT_PICKLE
    else:
        # In contrast, scikit-learn models saved in versions of MLflow > 1.9.1 do not
        # specify the ``data`` field within the pyfunc flavor configuration. For these newer
        # models, the ``path`` parameter of ``load_pyfunc()`` refers to the top-level MLflow
        # Model directory. In this case, we parse the model path from the MLmodel's pyfunc
        # flavor configuration and attempt to fetch the serialization format from the
        # scikit-learn flavor configuration
        try:
            sklearn_flavor_conf = _get_flavor_configuration(
                model_path=path, flavor_name=FLAVOR_NAME
            )
            serialization_format = sklearn_flavor_conf.get(
                "serialization_format", SERIALIZATION_FORMAT_PICKLE
            )
        except MlflowException:
            _logger.warning(
                "Could not find scikit-learn flavor configuration during model loading process."
                " Assuming 'pickle' serialization format."
            )
            serialization_format = SERIALIZATION_FORMAT_PICKLE

        pyfunc_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=pyfunc.FLAVOR_NAME
        )
        path = os.path.join(path, pyfunc_flavor_conf["model_path"])

    return _SklearnModelWrapper(
        _load_model_from_local_file(path=path, serialization_format=serialization_format)
    )


class _SklearnModelWrapper:
    _SUPPORTED_CUSTOM_PREDICT_FN = [
        "predict_proba",
        "predict_log_proba",
        "predict_joint_log_proba",
        "score",
    ]

    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model

        # Patch the model with custom predict functions that can be specified
        # via `pyfunc_predict_fn` argument when saving or logging.
        for predict_fn in self._SUPPORTED_CUSTOM_PREDICT_FN:
            if fn := getattr(self.sklearn_model, predict_fn, None):
                setattr(self, predict_fn, fn)

    def get_raw_model(self):
        """
        Returns the underlying scikit-learn model.
        """
        return self.sklearn_model

    def predict(
        self,
        data,
        params: dict[str, Any] | None = None,
    ):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        return self.sklearn_model.predict(data)


class _SklearnCustomModelPicklingError(pickle.PicklingError):
    """
    Exception for describing error raised during pickling custom sklearn estimator
    """

    def __init__(self, sk_model, original_exception):
        """
        Args:
            sk_model: The custom sklearn model to be pickled
            original_exception: The original exception raised
        """
        super().__init__(
            f"Pickling custom sklearn model {sk_model.__class__.__name__} failed "
            f"when saving model: {original_exception}"
        )
        self.original_exception = original_exception


def _dump_model(pickle_lib, sk_model, out):
    try:
        # Using python's default protocol to optimize compatibility.
        # Otherwise cloudpickle uses latest protocol leading to incompatibilities.
        # See https://github.com/mlflow/mlflow/issues/5419
        pickle_lib.dump(sk_model, out, protocol=pickle.DEFAULT_PROTOCOL)
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        if sk_model.__class__ not in _gen_estimators_to_patch():
            raise _SklearnCustomModelPicklingError(sk_model, e)
        else:
            raise


def _save_model(sk_model, output_path, serialization_format):
    """
    Args:
        sk_model: The scikit-learn model to serialize.
        output_path: The file path to which to write the serialized model.
        serialization_format: The format in which to serialize the model. This should be one of
            the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
            ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            _dump_model(pickle, sk_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            _dump_model(cloudpickle, sk_model, out)
        else:
            raise MlflowException(
                message=f"Unrecognized serialization format: {serialization_format}",
                error_code=INTERNAL_ERROR,
            )


def load_model(model_uri, dst_path=None):
    """
    Load a scikit-learn model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model, for example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A scikit-learn model.

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn

        sk_model = mlflow.sklearn.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/sk_models")

        # use Pandas DataFrame to make predictions
        pandas_df = ...
        predictions = sk_model.predict(pandas_df)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sklearn_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get("serialization_format", SERIALIZATION_FORMAT_PICKLE)
    return _load_model_from_local_file(
        path=sklearn_model_artifacts_path, serialization_format=serialization_format
    )


# The `_apis_autologging_disabled` contains APIs which is incompatible with autologging,
# when user call these APIs, autolog is temporarily disabled.
_apis_autologging_disabled = [
    "cross_validate",
    "cross_val_predict",
    "cross_val_score",
    "learning_curve",
    "permutation_test_score",
    "validation_curve",
]


class _AutologgingMetricsManager:
    """
    This class is designed for holding information which is used by autologging metrics
    It will hold information of:
    (1) a map of "prediction result object id" to a tuple of dataset name(the dataset is
       the one which generate the prediction result) and run_id.
       Note: We need this map instead of setting the run_id into the "prediction result object"
       because the object maybe a numpy array which does not support additional attribute
       assignment.
    (2) _log_post_training_metrics_enabled flag, in the following method scope:
       `model.fit` and `model.score`, in order to avoid nested/duplicated autologging metric, when
       run into these scopes, we need temporarily disable the metric autologging.
    (3) _eval_dataset_info_map, it is a double level map:
       `_eval_dataset_info_map[run_id][eval_dataset_var_name]` will get a list, each
       element in the list is an id of "eval_dataset" instance.
       This data structure is used for:
        * generating unique dataset name key when autologging metric. For each eval dataset object,
          if they have the same eval_dataset_var_name, but object ids are different,
          then they will be assigned different name (via appending index to the
          eval_dataset_var_name) when autologging.
    (4) _metric_api_call_info, it is a double level map:
       `_metric_api_call_info[run_id][metric_name]` will get a list of tuples, each tuple is:
         (logged_metric_key, metric_call_command_string)
        each call command string is like `metric_fn(arg1, arg2, ...)`
        This data structure is used for:
         * storing the call arguments dict for each metric call, we need log them into metric_info
           artifact file.

    Note: this class is not thread-safe.
    Design rule for this class:
     Because this class instance is a global instance, in order to prevent memory leak, it should
     only holds IDs and other small objects references. This class internal data structure should
     avoid reference to user dataset variables or model variables.
    """

    def __init__(self):
        self._pred_result_id_mapping = {}
        self._eval_dataset_info_map = defaultdict(lambda: defaultdict(list))
        self._metric_api_call_info = defaultdict(lambda: defaultdict(list))
        self._log_post_training_metrics_enabled = True
        self._metric_info_artifact_need_update = defaultdict(lambda: False)
        self._model_id_mapping = {}

    def should_log_post_training_metrics(self):
        """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safe guard checking,
        See following note.

        Note: It includes checking `_SklearnTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. GridSearchCV) case:
          running GridSearchCV.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
        return not _SklearnTrainingSession.is_active() and self._log_post_training_metrics_enabled

    def disable_log_post_training_metrics(self):
        class LogPostTrainingMetricsDisabledScope:
            def __enter__(inner_self):
                inner_self.old_status = self._log_post_training_metrics_enabled
                self._log_post_training_metrics_enabled = False

            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                self._log_post_training_metrics_enabled = inner_self.old_status

        return LogPostTrainingMetricsDisabledScope()

    @staticmethod
    def get_run_id_for_model(model):
        return getattr(model, "_mlflow_run_id", None)

    @staticmethod
    def is_metric_value_loggable(metric_value):
        """
        Check whether the specified `metric_value` is a numeric value which can be logged
        as an MLflow metric.
        """
        return isinstance(metric_value, (int, float, np.number)) and not isinstance(
            metric_value, bool
        )

    def register_model(self, model, run_id):
        """
        In `patched_fit`, we need register the model with the run_id used in `patched_fit`
        So that in following metric autologging, the metric will be logged into the registered
        run_id
        """
        model._mlflow_run_id = run_id

    def record_model_id(self, model, model_id):
        """
        Record the id(model) -> model_id mapping so that we can log metrics to the
        model later.
        """
        self._model_id_mapping[id(model)] = model_id

    def get_model_id_for_model(self, model) -> str | None:
        return self._model_id_mapping.get(id(model))

    @staticmethod
    def gen_name_with_index(name, index):
        assert index >= 0
        if index == 0:
            return name
        else:
            # Use '-' as the separator between name and index,
            # The '-' is not valid character in python var name
            # so it can prevent name conflicts after appending index.
            return f"{name}-{index + 1}"

    def register_prediction_input_dataset(self, model, eval_dataset):
        """
        Register prediction input dataset into eval_dataset_info_map, it will do:
         1. inspect eval dataset var name.
         2. check whether eval_dataset_info_map already registered this eval dataset.
            will check by object id.
         3. register eval dataset with id.
         4. return eval dataset name with index.

        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """
        eval_dataset_name = _inspect_original_var_name(
            eval_dataset, fallback_name="unknown_dataset"
        )
        eval_dataset_id = id(eval_dataset)

        run_id = self.get_run_id_for_model(model)
        registered_dataset_list = self._eval_dataset_info_map[run_id][eval_dataset_name]

        for i, id_i in enumerate(registered_dataset_list):
            if eval_dataset_id == id_i:
                index = i
                break
        else:
            index = len(registered_dataset_list)

        if index == len(registered_dataset_list):
            # register new eval dataset
            registered_dataset_list.append(eval_dataset_id)

        return self.gen_name_with_index(eval_dataset_name, index)

    def register_prediction_result(self, run_id, eval_dataset_name, predict_result, model_id=None):
        """
        Register the relationship
         id(prediction_result) --> (eval_dataset_name, run_id, model_id)
        into map `_pred_result_id_mapping`
        """
        value = (eval_dataset_name, run_id, model_id)
        prediction_result_id = id(predict_result)
        self._pred_result_id_mapping[prediction_result_id] = value

        def clean_id(id_):
            _AUTOLOGGING_METRICS_MANAGER._pred_result_id_mapping.pop(id_, None)

        # When the `predict_result` object being GCed, its ID may be reused, so register a finalizer
        # to clear the ID from the dict for preventing wrong ID mapping.
        weakref.finalize(predict_result, clean_id, prediction_result_id)

    @staticmethod
    def gen_metric_call_command(self_obj, metric_fn, *call_pos_args, **call_kwargs):
        """
        Generate metric function call command string like `metric_fn(arg1, arg2, ...)`
        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.

        Args:
            self_obj: If the metric_fn is a method of an instance (e.g. `model.score`),
                the `self_obj` represent the instance.
            metric_fn: metric function.
            call_pos_args: the positional arguments of the metric function call. If `metric_fn`
                is instance method, then the `call_pos_args` should exclude the first `self`
                argument.
            call_kwargs: the keyword arguments of the metric function call.
        """

        arg_list = []

        def arg_to_str(arg):
            if arg is None or np.isscalar(arg):
                if isinstance(arg, str) and len(arg) > 32:
                    # truncate too long string
                    return repr(arg[:32] + "...")
                return repr(arg)
            else:
                # dataset arguments or other non-scalar type argument
                return _inspect_original_var_name(arg, fallback_name=f"<{arg.__class__.__name__}>")

        param_sig = inspect.signature(metric_fn).parameters
        arg_names = list(param_sig.keys())

        if self_obj is not None:
            # If metric_fn is a method of an instance, e.g. `model.score`,
            # then the first argument is `self` which we need exclude it.
            arg_names.pop(0)

        if self_obj is not None:
            call_fn_name = f"{self_obj.__class__.__name__}.{metric_fn.__name__}"
        else:
            call_fn_name = metric_fn.__name__

        # Attach param signature key for positinal param values
        for arg_name, arg in zip(arg_names, call_pos_args):
            arg_list.append(f"{arg_name}={arg_to_str(arg)}")

        for arg_name, arg in call_kwargs.items():
            arg_list.append(f"{arg_name}={arg_to_str(arg)}")

        arg_list_str = ", ".join(arg_list)

        return f"{call_fn_name}({arg_list_str})"

    def register_metric_api_call(self, run_id, metric_name, dataset_name, call_command):
        """
        This method will do:
        (1) Generate and return metric key, format is:
          {metric_name}[-{call_index}]_{eval_dataset_name}
          metric_name is generated by metric function name, if multiple calls on the same
          metric API happen, the following calls will be assigned with an increasing "call index".
        (2) Register the metric key with the "call command" information into
          `_AUTOLOGGING_METRICS_MANAGER`. See doc of `gen_metric_call_command` method for
          details of "call command".
        """

        call_cmd_list = self._metric_api_call_info[run_id][metric_name]

        index = len(call_cmd_list)
        metric_name_with_index = self.gen_name_with_index(metric_name, index)
        metric_key = f"{metric_name_with_index}_{dataset_name}"

        call_cmd_list.append((metric_key, call_command))

        # Set the flag to true, represent the metric info in this run need update.
        # Later when `log_eval_metric` called, it will generate a new metric_info artifact
        # and overwrite the old artifact.
        self._metric_info_artifact_need_update[run_id] = True
        return metric_key

    def get_info_for_metric_api_call(self, call_pos_args, call_kwargs):
        """
        Given a metric api call (include the called metric function, and call arguments)
        Register the call information (arguments dict) into the `metric_api_call_arg_dict_list_map`
        and return a tuple of (run_id, eval_dataset_name, model_id)
        """
        call_arg_list = list(call_pos_args) + list(call_kwargs.values())

        dataset_id_list = self._pred_result_id_mapping.keys()

        # Note: some metric API the arguments is not like `y_true`, `y_pred`
        #  e.g.
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
        for arg in call_arg_list:
            if arg is not None and not np.isscalar(arg) and id(arg) in dataset_id_list:
                dataset_name, run_id, model_id = self._pred_result_id_mapping[id(arg)]
                break
        else:
            return None, None, None

        return run_id, dataset_name, model_id

    def log_post_training_metric(self, run_id, key, value, model_id=None):
        """
        Log the metric into the specified mlflow run.
        and it will also update the metric_info artifact if needed.
        If model_id is not None, metrics are logged into the model as well.
        """
        # Note: if the case log the same metric key multiple times,
        #  newer value will overwrite old value
        client = MlflowClient()
        client.log_metric(run_id=run_id, key=key, value=value, model_id=model_id)
        if self._metric_info_artifact_need_update[run_id]:
            call_commands_list = []
            for v in self._metric_api_call_info[run_id].values():
                call_commands_list.extend(v)

            call_commands_list.sort(key=lambda x: x[0])
            dict_to_log = OrderedDict(call_commands_list)
            client.log_dict(run_id=run_id, dictionary=dict_to_log, artifact_file="metric_info.json")
            self._metric_info_artifact_need_update[run_id] = False


# The global `_AutologgingMetricsManager` instance which holds information used in
# post-training metric autologging. See doc of class `_AutologgingMetricsManager` for details.
_AUTOLOGGING_METRICS_MANAGER = _AutologgingMetricsManager()


_metric_api_excluding_list = ["check_scoring", "get_scorer", "make_scorer", "get_scorer_names"]


def _get_metric_name_list():
    """
    Return metric function name list in `sklearn.metrics` module
    """
    from sklearn import metrics

    metric_list = []
    for metric_method_name in metrics.__all__:
        # excludes plot_* methods
        # exclude class (e.g. metrics.ConfusionMatrixDisplay)
        metric_method = getattr(metrics, metric_method_name)
        if (
            metric_method_name not in _metric_api_excluding_list
            and not inspect.isclass(metric_method)
            and callable(metric_method)
            and not metric_method_name.startswith("plot_")
        ):
            metric_list.append(metric_method_name)
    return metric_list


def _patch_estimator_method_if_available(
    flavor_name, class_def, func_name, patched_fn, manage_run, extra_tags=None
):
    if not hasattr(class_def, func_name):
        return

    original = gorilla.get_original_attribute(
        class_def, func_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        class_def, func_name, bypass_descriptor_protocol=True
    )
    if raw_original_obj == original and (callable(original) or isinstance(original, property)):
        # normal method or property decorated method
        safe_patch(
            flavor_name,
            class_def,
            func_name,
            patched_fn,
            manage_run=manage_run,
            extra_tags=extra_tags,
        )
    elif hasattr(raw_original_obj, "delegate_names") or hasattr(raw_original_obj, "check"):
        # sklearn delegated method
        safe_patch(
            flavor_name,
            raw_original_obj,
            "fn",
            patched_fn,
            manage_run=manage_run,
            extra_tags=extra_tags,
        )
    else:
        # unsupported method type. skip patching
        pass


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
    max_tuning_runs=5,
    log_post_training_metrics=True,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    pos_label=None,
    extra_tags=None,
):
    """
    Enables (or disables) and configures autologging for scikit-learn estimators.

    **When is autologging performed?**
      Autologging is performed when you call:

      - ``estimator.fit()``
      - ``estimator.fit_predict()``
      - ``estimator.fit_transform()``

    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.get_params(deep=True)``. Note that ``get_params``
          is called with ``deep=True``. This means when you fit a meta estimator that chains
          a series of estimators, the parameters of these child estimators are also logged.

      **Training metrics**
        - A training score obtained by ``estimator.score``. Note that the training score is
          computed using parameters given to ``fit()``.
        - Common metrics for classifier:

          - `precision score`_

          .. _precision score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

          - `recall score`_

          .. _recall score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

          - `f1 score`_

          .. _f1 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

          - `accuracy score`_

          .. _accuracy score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

          If the classifier has method ``predict_proba``, we additionally log:

          - `log loss`_

          .. _log loss:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

          - `roc auc score`_

          .. _roc auc score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

        - Common metrics for regressor:

          - `mean squared error`_

          .. _mean squared error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

          - root mean squared error

          - `mean absolute error`_

          .. _mean absolute error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

          - `r2 score`_

          .. _r2 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

      .. _post training metrics:

      **Post training metrics**
        When users call metric APIs after model training, MLflow tries to capture the metric API
        results and log them as MLflow metrics to the Run associated with the model. The following
        types of scikit-learn metric APIs are supported:

        - model.score
        - metric APIs defined in the `sklearn.metrics` module

        For post training metrics autologging, the metric key format is:
        "{metric_name}[-{call_index}]_{dataset_name}"

        - If the metric function is from `sklearn.metrics`, the MLflow "metric_name" is the
          metric function name. If the metric function is `model.score`, then "metric_name" is
          "{model_class_name}_score".
        - If multiple calls are made to the same scikit-learn metric API, each subsequent call
          adds a "call_index" (starting from 2) to the metric key.
        - MLflow uses the prediction input dataset variable name as the "dataset_name" in the
          metric key. The "prediction input dataset variable" refers to the variable which was
          used as the first argument of the associated `model.predict` or `model.score` call.
          Note: MLflow captures the "prediction input dataset" instance in the outermost call
          frame and fetches the variable name in the outermost call frame. If the "prediction
          input dataset" instance is an intermediate expression without a defined variable
          name, the dataset name is set to "unknown_dataset". If multiple "prediction input
          dataset" instances have the same variable name, then subsequent ones will append an
          index (starting from 2) to the inspected dataset name.

        **Limitations**
           - MLflow can only map the original prediction result object returned by a model
             prediction API (including predict / predict_proba / predict_log_proba / transform,
             but excluding fit_predict / fit_transform.) to an MLflow run.
             MLflow cannot find run information
             for other objects derived from a given prediction result (e.g. by copying or selecting
             a subset of the prediction result). scikit-learn metric APIs invoked on derived objects
             do not log metrics to MLflow.
           - Autologging must be enabled before scikit-learn metric APIs are imported from
             `sklearn.metrics`. Metric APIs imported before autologging is enabled do not log
             metrics to MLflow runs.
           - If user define a scorer which is not based on metric APIs in `sklearn.metrics`, then
             then post training metric autologging for the scorer is invalid.

        **Tags**
          - An estimator class name (e.g. "LinearRegression").
          - A fully qualified estimator class name
            (e.g. "sklearn.linear_model._base.LinearRegression").

        **Artifacts**
          - An MLflow Model with the :py:mod:`mlflow.sklearn` flavor containing a fitted estimator
            (logged by :py:func:`mlflow.sklearn.log_model()`). The Model also contains the
            :py:mod:`mlflow.pyfunc` flavor when the scikit-learn estimator defines `predict()`.
          - For post training metrics API calls, a "metric_info.json" artifact is logged. This is a
            JSON object whose keys are MLflow post training metric names
            (see "Post training metrics" section for the key format) and whose values are the
            corresponding metric call commands that produced the metrics, e.g.
            ``accuracy_score(y_true=test_iris_y, y_pred=pred_iris_y, normalize=False)``.

    **How does autologging work for meta estimators?**
      When a meta estimator (e.g. `Pipeline`_, `GridSearchCV`_) calls ``fit()``, it internally calls
      ``fit()`` on its child estimators. Autologging does NOT perform logging on these constituent
      ``fit()`` calls.

      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`GridSearchCV`_ and `RandomizedSearchCV`_) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model (if available).

    **Supported estimators**
      - All estimators obtained by `sklearn.utils.all_estimators`_ (including meta estimators).
      - `Pipeline`_
      - Parameter search estimators (`GridSearchCV`_ and `RandomizedSearchCV`_)

    .. _sklearn.utils.all_estimators:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.all_estimators.html

    .. _Pipeline:
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    .. _GridSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    .. _RandomizedSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    **Example**

    `See more examples <https://github.com/mlflow/mlflow/blob/master/examples/sklearn_autolog>`_

    .. code-block:: python

        from pprint import pprint
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import mlflow
        from mlflow import MlflowClient


        def fetch_logged_data(run_id):
            client = MlflowClient()
            data = client.get_run(run_id).data
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
            return data.params, data.metrics, tags, artifacts


        # enable autologging
        mlflow.sklearn.autolog()

        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        # train a model
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)

        # fetch logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

        pprint(params)
        # {'copy_X': 'True',
        #  'fit_intercept': 'True',
        #  'n_jobs': 'None',
        #  'normalize': 'False'}

        pprint(metrics)
        # {'training_score': 1.0,
        #  'training_mean_absolute_error': 2.220446049250313e-16,
        #  'training_mean_squared_error': 1.9721522630525295e-31,
        #  'training_r2_score': 1.0,
        #  'training_root_mean_squared_error': 4.440892098500626e-16}

        pprint(tags)
        # {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
        #  'estimator_name': 'LinearRegression'}

        pprint(artifacts)
        # ['model/MLmodel', 'model/conda.yaml', 'model/model.pkl']

    Args:
        log_input_examples: If ``True``, input examples from training datasets are collected and
            logged along with scikit-learn model artifacts during training. If
            ``False``, input examples are not logged.
            Note: Input examples are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along
            with scikit-learn model artifacts during training. If ``False``,
            signatures are not logged.
            Note: Model signatures are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
            Input examples and model signatures, which are attributes of MLflow models,
            are also omitted when ``log_models`` is ``False``.
        log_datasets: If ``True``, train and validation dataset information is logged to MLflow
            Tracking if applicable. If ``False``, dataset information is not logged.
        disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
            enables the scikit-learn autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            scikit-learn that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during scikit-learn
            autologging. If ``False``, show all events and warnings during scikit-learn
            autologging.
        max_tuning_runs: The maximum number of child MLflow runs created for hyperparameter
            search estimators. To create child runs for the best `k` results from
            the search, set `max_tuning_runs` to `k`. The default value is to track
            the best 5 search parameter sets. If `max_tuning_runs=None`, then
            a child run is created for each search parameter set. Note: The best k
            results is based on ordering in `rank_test_score`. In the case of
            multi-metric evaluation with a custom scorer, the first scorer's
            `rank_test_score_<scorer_name>` will be used to select the best k
            results. To change metric used for selecting best k results, change
            ordering of dict passed as `scoring` parameter for estimator.
        log_post_training_metrics: If ``True``, post training metrics are logged. Defaults to
            ``True``. See the `post training metrics`_ section for more
            details.
        serialization_format: The format in which to serialize the model. This should be one of
            the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
            ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        pos_label: If given, used as the positive label to compute binary classification
            training metrics such as precision, recall, f1, etc. This parameter should
            only be set for binary classification model. If used for multi-label model,
            the training metrics calculation will fail and the training metrics won't
            be logged. If used for regression model, the parameter will be ignored.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    _autolog(
        flavor_name=FLAVOR_NAME,
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
        log_models=log_models,
        log_datasets=log_datasets,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        max_tuning_runs=max_tuning_runs,
        log_post_training_metrics=log_post_training_metrics,
        serialization_format=serialization_format,
        pos_label=pos_label,
        extra_tags=extra_tags,
    )


def _autolog(
    flavor_name=FLAVOR_NAME,
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5,
    log_post_training_metrics=True,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    pos_label=None,
    extra_tags=None,
):
    """
    Internal autologging function for scikit-learn models.

    Args:
        flavor_name: A string value. Enable a ``mlflow.sklearn`` autologging routine
            for a flavor. By default it enables autologging for original
            scikit-learn models, as ``mlflow.sklearn.autolog()`` does. If
            the argument is `xgboost`, autologging for XGBoost scikit-learn
            models is enabled.
    """
    import pandas as pd
    import sklearn
    import sklearn.metrics
    import sklearn.model_selection

    from mlflow.models import infer_signature
    from mlflow.sklearn.utils import (
        _TRAINING_PREFIX,
        _create_child_runs_for_parameter_search,
        _gen_lightgbm_sklearn_estimators_to_patch,
        _gen_xgboost_sklearn_estimators_to_patch,
        _get_estimator_info_tags,
        _get_X_y_and_sample_weight,
        _is_parameter_search_estimator,
        _log_estimator_content,
        _log_parameter_search_results_as_artifact,
    )
    from mlflow.tracking.context import registry as context_registry

    if max_tuning_runs is not None and max_tuning_runs < 0:
        raise MlflowException(
            message=f"`max_tuning_runs` must be non-negative, instead got {max_tuning_runs}.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    def fit_mlflow_xgboost_and_lightgbm(original, self, *args, **kwargs):
        """
        Autologging function for XGBoost and LightGBM scikit-learn models
        """
        # Obtain a copy of a model input example from the training dataset prior to model training
        # for subsequent use during model logging, ensuring that the input example and inferred
        # model signature to not include any mutations from model training
        input_example_exc = None
        try:
            input_example = deepcopy(
                _get_X_y_and_sample_weight(self.fit, args, kwargs)[0][:INPUT_EXAMPLE_SAMPLE_ROWS]
            )
        except Exception as e:
            input_example_exc = e

        def get_input_example():
            if input_example_exc is not None:
                raise input_example_exc
            else:
                return input_example

        # parameter, metric, and non-model artifact logging are done in
        # `train()` in `mlflow.xgboost.autolog()` and `mlflow.lightgbm.autolog()`
        fit_output = original(self, *args, **kwargs)
        # log models after training
        if log_models:
            input_example, signature = resolve_input_example_and_signature(
                get_input_example,
                lambda input_example: infer_signature(
                    input_example,
                    # Copy the input example so that it is not mutated by the call to
                    # predict() prior to signature inference
                    self.predict(deepcopy(input_example)),
                ),
                log_input_examples,
                log_model_signatures,
                _logger,
            )
            log_model_func = (
                mlflow.xgboost.log_model
                if flavor_name == mlflow.xgboost.FLAVOR_NAME
                else mlflow.lightgbm.log_model
            )
            registered_model_name = get_autologging_config(
                flavor_name, "registered_model_name", None
            )
            if flavor_name == mlflow.xgboost.FLAVOR_NAME:
                model_format = get_autologging_config(flavor_name, "model_format", "ubj")
                model_info = log_model_func(
                    self,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    model_format=model_format,
                )
            else:
                model_info = log_model_func(
                    self,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                )
            _AUTOLOGGING_METRICS_MANAGER.record_model_id(self, model_info.model_id)
        return fit_output

    def fit_mlflow(original, self, *args, **kwargs):
        """
        Autologging function that performs model training by executing the training method
        referred to be `func_name` on the instance of `clazz` referred to by `self` & records
        MLflow parameters, metrics, tags, and artifacts to a corresponding MLflow Run.
        """
        # Obtain a copy of the training dataset prior to model training for subsequent
        # use during model logging & input example extraction, ensuring that we don't
        # attempt to infer input examples on data that was mutated during training
        (X, y_true, sample_weight) = _get_X_y_and_sample_weight(self.fit, args, kwargs)
        autologging_client = MlflowAutologgingQueueingClient()
        _log_pretraining_metadata(autologging_client, self, X, y_true)
        params_logging_future = autologging_client.flush(synchronous=False)
        fit_output = original(self, *args, **kwargs)
        _log_posttraining_metadata(autologging_client, self, X, y_true, sample_weight)
        autologging_client.flush(synchronous=True)
        params_logging_future.await_completion()
        return fit_output

    def _log_pretraining_metadata(autologging_client, estimator, X, y):
        """
        Records metadata (e.g., params and tags) for a scikit-learn estimator prior to training.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.

        Args:
            autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
                efficiently logging run data to MLflow Tracking.
            estimator: The scikit-learn estimator for which to log metadata.
        """
        # Deep parameter logging includes parameters from children of a given
        # estimator. For some meta estimators (e.g., pipelines), recording
        # these parameters is desirable. For parameter search estimators,
        # however, child estimators act as seeds for the parameter search
        # process; accordingly, we avoid logging initial, untuned parameters
        # for these seed estimators.
        should_log_params_deeply = not _is_parameter_search_estimator(estimator)
        run_id = mlflow.active_run().info.run_id
        autologging_client.log_params(
            run_id=mlflow.active_run().info.run_id,
            params=estimator.get_params(deep=should_log_params_deeply),
        )
        autologging_client.set_tags(
            run_id=run_id,
            tags=_get_estimator_info_tags(estimator),
        )

        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(context_tags)

                if dataset := _create_dataset(X, source, y):
                    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="train")]
                    dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)

                    autologging_client.log_inputs(
                        run_id=mlflow.active_run().info.run_id, datasets=[dataset_input]
                    )
            except Exception as e:
                _logger.warning(
                    "Failed to log training dataset information to MLflow Tracking. Reason: %s", e
                )

    def _log_posttraining_metadata(autologging_client, estimator, X, y, sample_weight):
        """
        Records metadata for a scikit-learn estimator after training has completed.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.

        Args:
            autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
                efficiently logging run data to MLflow Tracking.
            estimator: The scikit-learn estimator for which to log metadata.
            X: The training dataset samples passed to the ``estimator.fit()`` function.
            y: The training dataset labels passed to the ``estimator.fit()`` function.
            sample_weight: Sample weights passed to the ``estimator.fit()`` function.
        """
        # Fetch an input example using the first several rows of the array-like
        # training data supplied to the training routine (e.g., `fit()`). Copy the
        # example to avoid mutation during subsequent metric computations
        input_example_exc = None
        try:
            input_example = deepcopy(X[:INPUT_EXAMPLE_SAMPLE_ROWS])
        except Exception as e:
            input_example_exc = e

        def get_input_example():
            if input_example_exc is not None:
                raise input_example_exc
            else:
                return input_example

        def infer_model_signature(input_example):
            if hasattr(estimator, "predict"):
                # Copy the input example so that it is not mutated by the call to
                # predict() prior to signature inference
                model_output = estimator.predict(deepcopy(input_example))
            elif hasattr(estimator, "transform"):
                model_output = estimator.transform(deepcopy(input_example))
            else:
                raise Exception(
                    "the trained model does not have a `predict` or `transform` "
                    "function, which is required in order to infer the signature"
                )

            return infer_signature(input_example, model_output)

        def _log_model_with_except_handling(*args, **kwargs):
            try:
                return log_model(*args, **kwargs)
            except _SklearnCustomModelPicklingError as e:
                _logger.warning(str(e))

        model_id = None
        if log_models:
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
            should_log_params_deeply = not _is_parameter_search_estimator(estimator)
            params = estimator.get_params(deep=should_log_params_deeply)
            if hasattr(estimator, "best_params_"):
                params |= {
                    f"best_{param_name}": param_value
                    for param_name, param_value in estimator.best_params_.items()
                }
            if logged_model := _log_model_with_except_handling(
                estimator,
                name="model",
                signature=signature,
                input_example=input_example,
                serialization_format=serialization_format,
                registered_model_name=registered_model_name,
                params=params,
            ):
                model_id = logged_model.model_id
                _AUTOLOGGING_METRICS_MANAGER.record_model_id(estimator, logged_model.model_id)

        # log common metrics and artifacts for estimators (classifier, regressor)
        context_tags = context_registry.resolve_tags()
        source = CodeDatasetSource(context_tags)
        try:
            dataset = _create_dataset(X, source, y)
        except Exception:
            _logger.debug("Failed to create dataset for logging.", exc_info=True)
            dataset = None
        logged_metrics = _log_estimator_content(
            autologging_client=autologging_client,
            estimator=estimator,
            prefix=_TRAINING_PREFIX,
            run_id=mlflow.active_run().info.run_id,
            X=X,
            y_true=y,
            sample_weight=sample_weight,
            pos_label=pos_label,
            dataset=dataset,
            model_id=model_id,
        )
        if y is None and not logged_metrics:
            _logger.warning(
                "Training metrics will not be recorded because training labels were not specified."
                " To automatically record training metrics, provide training labels as inputs to"
                " the model training function."
            )

        best_estimator_model_id = None
        best_estimator_params = None
        if _is_parameter_search_estimator(estimator):
            if hasattr(estimator, "best_estimator_") and log_models:
                best_estimator_params = estimator.best_estimator_.get_params(deep=True)
                if model_info := _log_model_with_except_handling(
                    estimator.best_estimator_,
                    name="best_estimator",
                    signature=signature,
                    input_example=input_example,
                    serialization_format=serialization_format,
                    params=best_estimator_params,
                ):
                    best_estimator_model_id = model_info.model_id

            if hasattr(estimator, "best_score_"):
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics={"best_cv_score": estimator.best_score_},
                    dataset=dataset,
                    model_id=model_id,
                )

            if hasattr(estimator, "best_params_"):
                best_params = {
                    f"best_{param_name}": param_value
                    for param_name, param_value in estimator.best_params_.items()
                }
                autologging_client.log_params(
                    run_id=mlflow.active_run().info.run_id,
                    params=best_params,
                )

            if hasattr(estimator, "cv_results_"):
                try:
                    # Fetch environment-specific tags (e.g., user and source) to ensure that lineage
                    # information is consistent with the parent run
                    child_tags = context_registry.resolve_tags()
                    child_tags.update({MLFLOW_AUTOLOGGING: flavor_name})
                    _create_child_runs_for_parameter_search(
                        autologging_client=autologging_client,
                        cv_estimator=estimator,
                        parent_run=mlflow.active_run(),
                        max_tuning_runs=max_tuning_runs,
                        child_tags=child_tags,
                        dataset=dataset,
                        best_estimator_params=best_estimator_params,
                        best_estimator_model_id=best_estimator_model_id,
                    )
                except Exception as e:
                    _logger.warning(
                        "Encountered exception during creation of child runs for parameter search."
                        f" Child runs may be missing. Exception: {e}"
                    )

                try:
                    cv_results_df = pd.DataFrame.from_dict(estimator.cv_results_)
                    _log_parameter_search_results_as_artifact(
                        cv_results_df, mlflow.active_run().info.run_id
                    )
                except Exception as e:
                    _logger.warning(
                        f"Failed to log parameter search results as an artifact. Exception: {e}"
                    )

    def patched_fit(fit_impl, allow_children_patch, original, self, *args, **kwargs):
        """
        Autologging patch function to be applied to a sklearn model class that defines a `fit`
        method and inherits from `BaseEstimator` (thereby defining the `get_params()` method)

        Args:
            fit_impl: The patched fit function implementation, the function should be defined as
                `fit_mlflow(original, self, *args, **kwargs)`, the `original` argument
                refers to the original `EstimatorClass.fit` method, the `self` argument
                refers to the estimator instance being patched, the `*args` and
                `**kwargs` are arguments passed to the original fit method.
            allow_children_patch: Whether to allow children sklearn session logging or not.
            original: the original `EstimatorClass.fit` method to be patched.
            self: the estimator instance being patched.
            args: positional arguments to be passed to the original fit method.
            kwargs: keyword arguments to be passed to the original fit method.
        """
        should_log_post_training_metrics = (
            log_post_training_metrics
            and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
        )

        with _SklearnTrainingSession(estimator=self, allow_children=allow_children_patch) as t:
            if t.should_log():
                # In `fit_mlflow` call, it will also call metric API for computing training metrics
                # so we need temporarily disable the post_training_metrics patching.
                with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                    result = fit_impl(original, self, *args, **kwargs)
                if should_log_post_training_metrics:
                    _AUTOLOGGING_METRICS_MANAGER.register_model(
                        self, mlflow.active_run().info.run_id
                    )
                return result
            else:
                return original(self, *args, **kwargs)

    def patched_predict(original, self, *args, **kwargs):
        """
        In `patched_predict`, register the prediction result instance with the run id and
         eval dataset name. e.g.
        ```
        prediction_result = model_1.predict(eval_X)
        ```
        then we need register the following relationship into the `_AUTOLOGGING_METRICS_MANAGER`:
        id(prediction_result) --> (eval_dataset_name, run_id)

        Note: we cannot set additional attributes "eval_dataset_name" and "run_id" into
        the prediction_result object, because certain dataset type like numpy does not support
        additional attribute assignment.
        """
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            # Avoid nested patch when nested inference calls happens.
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                predict_result = original(self, *args, **kwargs)
            eval_dataset = get_instance_method_first_arg_value(original, args, kwargs)
            eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(
                self, eval_dataset
            )
            _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(
                run_id,
                eval_dataset_name,
                predict_result,
                model_id=_AUTOLOGGING_METRICS_MANAGER.get_model_id_for_model(self),
            )
            if log_datasets:
                try:
                    context_tags = context_registry.resolve_tags()
                    source = CodeDatasetSource(context_tags)

                    # log the dataset
                    if dataset := _create_dataset(eval_dataset, source):
                        tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]
                        dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)

                        # log the dataset
                        client = mlflow.MlflowClient()
                        client.log_inputs(run_id=run_id, datasets=[dataset_input])
                except Exception as e:
                    _logger.warning(
                        "Failed to log evaluation dataset information to "
                        "MLflow Tracking. Reason: %s",
                        e,
                    )
            return predict_result
        else:
            return original(self, *args, **kwargs)

    def patched_metric_api(original, *args, **kwargs):
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics():
            # one metric api may call another metric api,
            # to avoid this, call disable_log_post_training_metrics to avoid nested patch
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                metric = original(*args, **kwargs)

            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(metric):
                metric_name = original.__name__
                call_command = _AUTOLOGGING_METRICS_MANAGER.gen_metric_call_command(
                    None, original, *args, **kwargs
                )

                (run_id, dataset_name, model_id) = (
                    _AUTOLOGGING_METRICS_MANAGER.get_info_for_metric_api_call(args, kwargs)
                )
                if run_id and dataset_name:
                    metric_key = _AUTOLOGGING_METRICS_MANAGER.register_metric_api_call(
                        run_id, metric_name, dataset_name, call_command
                    )
                    _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(
                        run_id, metric_key, metric, model_id=model_id
                    )

            return metric
        else:
            return original(*args, **kwargs)

    # we need patch model.score method because:
    #  some model.score() implementation won't call metric APIs in `sklearn.metrics`
    #  e.g.
    #  https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/covariance/_empirical_covariance.py#L220
    def patched_model_score(original, self, *args, **kwargs):
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            # `model.score` may call metric APIs internally, in order to prevent nested metric call
            # being logged, temporarily disable post_training_metrics patching.
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                score_value = original(self, *args, **kwargs)

            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(score_value):
                metric_name = f"{self.__class__.__name__}_score"
                call_command = _AUTOLOGGING_METRICS_MANAGER.gen_metric_call_command(
                    self, original, *args, **kwargs
                )

                eval_dataset = get_instance_method_first_arg_value(original, args, kwargs)
                eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(
                    self, eval_dataset
                )
                metric_key = _AUTOLOGGING_METRICS_MANAGER.register_metric_api_call(
                    run_id, metric_name, eval_dataset_name, call_command
                )
                model_id = _AUTOLOGGING_METRICS_MANAGER.get_model_id_for_model(self)
                _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(
                    run_id, metric_key, score_value, model_id=model_id
                )

            return score_value
        else:
            return original(self, *args, **kwargs)

    def _apply_sklearn_descriptor_unbound_method_call_fix():
        import sklearn

        if Version(sklearn.__version__) <= Version("0.24.2"):
            import sklearn.utils.metaestimators

            if not hasattr(sklearn.utils.metaestimators, "_IffHasAttrDescriptor"):
                return

            def patched_IffHasAttrDescriptor__get__(self, obj, type=None):
                """
                For sklearn version <= 0.24.2, `_IffHasAttrDescriptor.__get__` method does not
                support unbound method call.
                See https://github.com/scikit-learn/scikit-learn/issues/20614
                This patched function is for hot patch.
                """

                # raise an AttributeError if the attribute is not present on the object
                if obj is not None:
                    # delegate only on instances, not the classes.
                    # this is to allow access to the docstrings.
                    for delegate_name in self.delegate_names:
                        try:
                            delegate = sklearn.utils.metaestimators.attrgetter(delegate_name)(obj)
                        except AttributeError:
                            continue
                        else:
                            getattr(delegate, self.attribute_name)
                            break
                    else:
                        sklearn.utils.metaestimators.attrgetter(self.delegate_names[-1])(obj)

                    def out(*args, **kwargs):
                        return self.fn(obj, *args, **kwargs)

                else:
                    # This makes it possible to use the decorated method as an unbound method,
                    # for instance when monkeypatching.
                    def out(*args, **kwargs):
                        return self.fn(*args, **kwargs)

                # update the docstring of the returned function
                functools.update_wrapper(out, self.fn)
                return out

            update_wrapper_extended(
                patched_IffHasAttrDescriptor__get__,
                sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__,
            )

            sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__ = (
                patched_IffHasAttrDescriptor__get__
            )

    _apply_sklearn_descriptor_unbound_method_call_fix()

    if flavor_name == mlflow.xgboost.FLAVOR_NAME:
        estimators_to_patch = _gen_xgboost_sklearn_estimators_to_patch()
        patched_fit_impl = fit_mlflow_xgboost_and_lightgbm
        allow_children_patch = True
    elif flavor_name == mlflow.lightgbm.FLAVOR_NAME:
        estimators_to_patch = _gen_lightgbm_sklearn_estimators_to_patch()
        patched_fit_impl = fit_mlflow_xgboost_and_lightgbm
        allow_children_patch = True
    else:
        estimators_to_patch = _gen_estimators_to_patch()
        patched_fit_impl = fit_mlflow
        allow_children_patch = False

    for class_def in estimators_to_patch:
        # Patch fitting methods
        for func_name in ["fit", "fit_transform", "fit_predict"]:
            _patch_estimator_method_if_available(
                flavor_name,
                class_def,
                func_name,
                functools.partial(patched_fit, patched_fit_impl, allow_children_patch),
                manage_run=True,
                extra_tags=extra_tags,
            )

        # Patch inference methods
        for func_name in ["predict", "predict_proba", "transform", "predict_log_proba"]:
            _patch_estimator_method_if_available(
                flavor_name,
                class_def,
                func_name,
                patched_predict,
                manage_run=False,
            )

        # Patch scoring methods
        _patch_estimator_method_if_available(
            flavor_name,
            class_def,
            "score",
            patched_model_score,
            manage_run=False,
            extra_tags=extra_tags,
        )

    if log_post_training_metrics:
        for metric_name in _get_metric_name_list():
            safe_patch(
                flavor_name, sklearn.metrics, metric_name, patched_metric_api, manage_run=False
            )

        # `sklearn.metrics.SCORERS` was removed in scikit-learn 1.3
        if hasattr(sklearn.metrics, "get_scorer_names"):
            for scoring in sklearn.metrics.get_scorer_names():
                scorer = sklearn.metrics.get_scorer(scoring)
                safe_patch(flavor_name, scorer, "_score_func", patched_metric_api, manage_run=False)
        else:
            for scorer in sklearn.metrics.SCORERS.values():
                safe_patch(flavor_name, scorer, "_score_func", patched_metric_api, manage_run=False)

    def patched_fn_with_autolog_disabled(original, *args, **kwargs):
        with disable_autologging():
            return original(*args, **kwargs)

    for disable_autolog_func_name in _apis_autologging_disabled:
        safe_patch(
            flavor_name,
            sklearn.model_selection,
            disable_autolog_func_name,
            patched_fn_with_autolog_disabled,
            manage_run=False,
        )

    def _create_dataset(X, source, y=None, dataset_name=None):
        # create a dataset
        from scipy.sparse import issparse

        if isinstance(X, pd.DataFrame):
            dataset = from_pandas(df=X, source=source)
        elif issparse(X):
            arr_X = X.toarray()
            if y is not None:
                dataset = from_numpy(
                    features=arr_X,
                    targets=y.toarray() if issparse(y) else y,
                    source=source,
                    name=dataset_name,
                )
            else:
                dataset = from_numpy(features=arr_X, source=source, name=dataset_name)
        elif isinstance(X, np.ndarray):
            if y is not None:
                dataset = from_numpy(features=X, targets=y, source=source, name=dataset_name)
            else:
                dataset = from_numpy(features=X, source=source, name=dataset_name)
        else:
            _logger.warning("Unrecognized dataset type %s. Dataset logging skipped.", type(X))
            return None
        return dataset
