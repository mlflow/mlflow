"""
The ``mlflow.statsmodels`` module provides an API for logging and loading statsmodels models.
This module exports statsmodels models with the following flavors:

statsmodels (native) format
    This is the main flavor that can be loaded back into statsmodels, which relies on pickle
    internally to serialize a model.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _statsmodels.base.model.Results:
    https://www.statsmodels.org/stable/_modules/statsmodels/base/model.html#Results

"""
import inspect
import itertools
import logging
import os
from typing import Any, Dict, Optional

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    log_fn_args_as_params,
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
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.validation import _is_numeric

FLAVOR_NAME = "statsmodels"
STATSMODELS_DATA_SUBPATH = "model.statsmodels"


_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("statsmodels")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


_model_size_threshold_for_emitting_warning = 100 * 1024 * 1024  # 100 MB


_save_model_called_from_autolog = False


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    statsmodels_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    remove_data: bool = False,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save a statsmodels model to a path on the local file system.

    Args:
        statsmodels_model: statsmodels model (an instance of `statsmodels.base.model.Results`_) to
            be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        remove_data: bool. If False (default), then the instance is pickled without changes. If
            True, then all arrays with length nobs are set to None before pickling. See the
            remove_data method. In some cases not all arrays will be set to None.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
    """
    import statsmodels

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    model_data_path = os.path.join(path, STATSMODELS_DATA_SUBPATH)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _StatsmodelsModelWrapper(statsmodels_model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    # Save a statsmodels model
    statsmodels_model.save(model_data_path, remove_data)
    if _save_model_called_from_autolog and not remove_data:
        saved_model_size = os.path.getsize(model_data_path)
        if saved_model_size >= _model_size_threshold_for_emitting_warning:
            _logger.warning(
                "The fitted model is larger than "
                f"{_model_size_threshold_for_emitting_warning // (1024 * 1024)} MB, "
                f"saving it as artifacts is time consuming.\n"
                "To reduce model size, use `mlflow.statsmodels.autolog(log_models=False)` and "
                "manually log model by "
                '`mlflow.statsmodels.log_model(model, remove_data=True, artifact_path="model")`'
            )

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.statsmodels",
        data=STATSMODELS_DATA_SUBPATH,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        statsmodels_version=statsmodels.__version__,
        data=STATSMODELS_DATA_SUBPATH,
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
    statsmodels_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    remove_data: bool = False,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Log a statsmodels model as an MLflow artifact for the current run.

    Args:
        statsmodels_model: statsmodels model (an instance of `statsmodels.base.model.Results`_) to
            be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under ``registered_model_name``,
            also creating a registered model if one with the given name does not exist.
        remove_data: bool. If False (default), then the instance is pickled without changes. If
            True, then all arrays with length nobs are set to None before pickling. See the
            remove_data method. In some cases not all arrays will be set to None.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish being
            created and is in ``READY`` status. By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the metadata
        of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.statsmodels,
        registered_model_name=registered_model_name,
        statsmodels_model=statsmodels_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        remove_data=remove_data,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def _load_model(path):
    import statsmodels.iolib.api as smio

    return smio.load_pickle(path)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``statsmodels`` flavor.
    """
    return _StatsmodelsModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """
    Load a statsmodels model from a local file or a run.

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
        A statsmodels model (an instance of `statsmodels.base.model.Results`_).

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    statsmodels_model_file_path = os.path.join(
        local_model_path, flavor_conf.get("data", STATSMODELS_DATA_SUBPATH)
    )
    return _load_model(path=statsmodels_model_file_path)


class _StatsmodelsModelWrapper:
    def __init__(self, statsmodels_model):
        self.statsmodels_model = statsmodels_model

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.statsmodels_model

    def predict(
        self,
        dataframe,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        from statsmodels.tsa.base.tsa_model import TimeSeriesModel

        model = self.statsmodels_model.model
        if isinstance(model, TimeSeriesModel):
            # Assume the inference dataframe has columns "start" and "end", and just one row
            # TODO: move this to a specific mlflow.statsmodels.tsa flavor? Time series models
            # often expect slightly different arguments to make predictions
            if dataframe.shape[0] != 1 or not (
                "start" in dataframe.columns and "end" in dataframe.columns
            ):
                raise MlflowException(
                    "prediction dataframes for a TimeSeriesModel must have exactly one row"
                    + " and include columns called start and end"
                )

            start_date = dataframe["start"][0]
            end_date = dataframe["end"][0]
            return self.statsmodels_model.predict(start=start_date, end=end_date)
        else:
            return self.statsmodels_model.predict(dataframe)


class AutologHelpers:
    # Autologging should be done only in the fit function called by the user, but not
    # inside other internal fit functions
    should_autolog = True


# Currently we only autolog basic metrics
_autolog_metric_allowlist = [
    "aic",
    "bic",
    "centered_tss",
    "condition_number",
    "df_model",
    "df_resid",
    "ess",
    "f_pvalue",
    "fvalue",
    "llf",
    "mse_model",
    "mse_resid",
    "mse_total",
    "rsquared",
    "rsquared_adj",
    "scale",
    "ssr",
    "uncentered_tss",
]


def _get_autolog_metrics(fitted_model):
    result_metrics = {}

    failed_evaluating_metrics = set()
    for metric in _autolog_metric_allowlist:
        try:
            if hasattr(fitted_model, metric):
                metric_value = getattr(fitted_model, metric)
                if _is_numeric(metric_value):
                    result_metrics[metric] = metric_value
        except Exception:
            failed_evaluating_metrics.add(metric)

    if len(failed_evaluating_metrics) > 0:
        _logger.warning(
            f"Failed to autolog metrics: {', '.join(sorted(failed_evaluating_metrics))}."
        )
    return result_metrics


@autologging_integration(FLAVOR_NAME)
def autolog(
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
    Enables (or disables) and configures automatic logging from statsmodels to MLflow.
    Logs the following:

    - allowlisted metrics returned by method `fit` of any subclass of
      statsmodels.base.model.Model, the allowlisted metrics including: {autolog_metric_allowlist}
    - trained model.
    - an html artifact which shows the model summary.

    Args:
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
            Input examples and model signatures, which are attributes of MLflow models,
            are also omitted when ``log_models`` is ``False``.
        log_datasets: If ``True``, dataset information is logged to MLflow Tracking.
            If ``False``, dataset information is not logged.
        disable: If ``True``, disables the statsmodels autologging integration. If ``False``,
            enables the statsmodels autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            statsmodels that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during statsmodels
            autologging. If ``False``, show all events and warnings during statsmodels
            autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    import statsmodels

    # Autologging depends on the exploration of the models class tree within the
    # `statsmodels.base.models` module. In order to load / access this module, the
    # `statsmodels.api` module must be imported
    import statsmodels.api

    def find_subclasses(klass):
        """
        Recursively return a (non-nested) list of the class object and all its subclasses.

        Args:
            klass: The class whose class subtree we want to retrieve.

        Returns:
            A list of classes that includes the argument in the first position.
        """
        subclasses = klass.__subclasses__()
        if subclasses:
            subclass_lists = [find_subclasses(c) for c in subclasses]
            chain = itertools.chain.from_iterable(subclass_lists)
            return [klass] + list(chain)
        else:
            return [klass]

    def overrides(klass, function_name):
        """
        Returns True when the class passed as first argument overrides the function_name
        Based on https://stackoverflow.com/a/62303206/5726057

        Args:
            klass: The class we are inspecting.
            function_name: A string with the name of the method we want to check overriding.

        Returns:
            True if the class overrides the function_name, False otherwise.
        """
        try:
            superclass = inspect.getmro(klass)[1]
            return getattr(klass, function_name) is not getattr(superclass, function_name)
        except (IndexError, AttributeError):
            return False

    def patch_class_tree(klass):
        """
        Patches all subclasses that override any auto-loggable method via monkey patching using
        the gorilla package, taking the argument as the tree root in the class hierarchy. Every
        auto-loggable method found in any of the subclasses is replaced by the patched version.

        Args:
            klass: Root in the class hierarchy to be analyzed and patched recursively.
        """

        # TODO: add more autologgable methods here (e.g. fit_regularized, from_formula, etc)
        # See https://www.statsmodels.org/dev/api.html
        autolog_supported_func = {"fit": wrapper_fit}
        glob_subclasses = set(find_subclasses(klass))

        # Create a patch for every method that needs to be patched, i.e. those
        # which actually override an autologgable method
        patches_list = [
            # Link the patched function with the original via a local variable in the closure
            # to allow invoking superclass methods in the context of the subclass, and not
            # losing the trace of the true original method
            (clazz, method_name, wrapper_func)
            for clazz in glob_subclasses
            for (method_name, wrapper_func) in autolog_supported_func.items()
            if overrides(clazz, method_name)
        ]

        for clazz, method_name, patch_impl in patches_list:
            safe_patch(
                FLAVOR_NAME, clazz, method_name, patch_impl, manage_run=True, extra_tags=extra_tags
            )

    def wrapper_fit(original, self, *args, **kwargs):
        should_autolog = False
        if AutologHelpers.should_autolog:
            AutologHelpers.should_autolog = False
            should_autolog = True

        try:
            if should_autolog:
                # This may generate warnings due to collisions in already-logged param names
                log_fn_args_as_params(original, args, kwargs)

            # training model
            model = original(self, *args, **kwargs)

            if should_autolog:
                # Log the model
                if get_autologging_config(FLAVOR_NAME, "log_models", True):
                    global _save_model_called_from_autolog
                    _save_model_called_from_autolog = True
                    registered_model_name = get_autologging_config(
                        FLAVOR_NAME, "registered_model_name", None
                    )
                    try:
                        log_model(
                            model,
                            artifact_path="model",
                            registered_model_name=registered_model_name,
                        )
                    finally:
                        _save_model_called_from_autolog = False

                # Log the most common metrics
                if isinstance(model, statsmodels.base.wrapper.ResultsWrapper):
                    metrics_dict = _get_autolog_metrics(model)
                    mlflow.log_metrics(metrics_dict)

                    model_summary = model.summary().as_text()
                    mlflow.log_text(model_summary, "model_summary.txt")

            return model

        finally:
            # Clean the shared flag for future calls in case it had been set here ...
            if should_autolog:
                AutologHelpers.should_autolog = True

    patch_class_tree(statsmodels.base.model.Model)


if autolog.__doc__ is not None:
    autolog.__doc__ = autolog.__doc__.format(
        autolog_metric_allowlist=", ".join(_autolog_metric_allowlist)
    )
