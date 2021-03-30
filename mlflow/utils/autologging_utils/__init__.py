# pylint: disable=unused-wildcard-import

import importlib
import inspect
import logging
import time
import contextlib
import yaml
from distutils.version import LooseVersion
from pkg_resources import resource_filename

import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH

# Define the module-level logger for autologging utilities before importing utilities
# defined in submodules (e.g., `safety`, `events`) that depend on the module-level logger
_logger = logging.getLogger(__name__)

# Import autologging utilities used by this module
from mlflow.utils.autologging_utils.logging_and_warnings import (
    set_mlflow_events_and_warnings_behavior_globally,
    set_non_mlflow_warnings_behavior_for_current_thread,
)
from mlflow.utils.autologging_utils.safety import (
    try_mlflow_log,
    update_wrapper_extended,
)

# Wildcard import other autologging utilities (e.g. safety utilities, event logging utilities) used
# in autologging integration implementations, which reference them via the
# `mlflow.utils.autologging_utils` module
from mlflow.utils.autologging_utils.safety import *
from mlflow.utils.autologging_utils.events import *


INPUT_EXAMPLE_SAMPLE_ROWS = 5
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)
_AUTOLOGGING_TEST_MODE_ENV_VAR = "MLFLOW_AUTOLOGGING_TESTING"

# Dict mapping integration name to its config.
AUTOLOGGING_INTEGRATIONS = {}

_logger = logging.getLogger(__name__)


def log_fn_args_as_params(fn, args, kwargs, unlogged=[]):  # pylint: disable=W0102
    """
    Log parameters explicitly passed to a function.

    :param fn: function whose parameters are to be logged
    :param args: arguments explicitly passed into fn. If `fn` is defined on a class,
                 `self` should not be part of `args`; the caller is responsible for
                 filtering out `self` before calling this function.
    :param kwargs: kwargs explicitly passed into fn
    :param unlogged: parameters not to be logged
    :return: None
    """
    param_spec = inspect.signature(fn).parameters
    # Filter out `self` from the signature under the assumption that it is not contained
    # within the specified `args`, as stipulated by the documentation
    relevant_params = [param for param in param_spec.values() if param.name != "self"]

    # Fetch the parameter names for specified positional arguments from the function
    # signature & create a mapping from positional argument name to specified value
    params_to_log = {
        param_info.name: param_val
        for param_info, param_val in zip(list(relevant_params)[: len(args)], args)
    }
    # Add all user-specified keyword arguments to the set of parameters to log
    params_to_log.update(kwargs)
    # Add parameters that were not explicitly specified by the caller to the mapping,
    # using their default values
    params_to_log.update(
        {
            param.name: param.default
            for param in list(relevant_params)[len(args) :]
            if param.name not in kwargs
        }
    )
    # Filter out any parameters that should not be logged, as specified by the `unlogged` parameter
    params_to_log = {key: value for key, value in params_to_log.items() if key not in unlogged}
    try_mlflow_log(mlflow.log_params, params_to_log)


class InputExampleInfo:
    """
    Stores info about the input example collection before it is needed.

    For example, in xgboost and lightgbm, an InputExampleInfo object is attached to the dataset,
    where its value is read later by the train method.

    Exactly one of input_example or error_msg should be populated.
    """

    def __init__(self, input_example=None, error_msg=None):
        self.input_example = input_example
        self.error_msg = error_msg


def resolve_input_example_and_signature(
    get_input_example, infer_model_signature, log_input_example, log_model_signature, logger
):
    """
    Handles the logic of calling functions to gather the input example and infer the model
    signature.

    :param get_input_example: function which returns an input example, usually sliced from a
                              dataset. This function can raise an exception, its message will be
                              shown to the user in a warning in the logs.
    :param infer_model_signature: function which takes an input example and returns the signature
                                  of the inputs and outputs of the model. This function can raise
                                  an exception, its message will be shown to the user in a warning
                                  in the logs.
    :param log_input_example: whether to log errors while collecting the input example, and if it
                              succeeds, whether to return the input example to the user. We collect
                              it even if this parameter is False because it is needed for inferring
                              the model signature.
    :param log_model_signature: whether to infer and return the model signature.
    :param logger: the logger instance used to log warnings to the user during input example
                   collection and model signature inference.

    :return: A tuple of input_example and signature. Either or both could be None based on the
             values of log_input_example and log_model_signature.
    """

    input_example = None
    input_example_user_msg = None
    input_example_failure_msg = None
    if log_input_example or log_model_signature:
        try:
            input_example = get_input_example()
        except Exception as e:
            input_example_failure_msg = str(e)
            input_example_user_msg = "Failed to gather input example: " + str(e)

    model_signature = None
    model_signature_user_msg = None
    if log_model_signature:
        try:
            if input_example is None:
                raise Exception(
                    "could not sample data to infer model signature: " + input_example_failure_msg
                )
            model_signature = infer_model_signature(input_example)
        except Exception as e:
            model_signature_user_msg = "Failed to infer model signature: " + str(e)

    if log_input_example and input_example_user_msg is not None:
        logger.warning(input_example_user_msg)
    if log_model_signature and model_signature_user_msg is not None:
        logger.warning(model_signature_user_msg)

    return input_example if log_input_example else None, model_signature


class BatchMetricsLogger:
    """
    The BatchMetricsLogger will log metrics in batch against an mlflow run.
    If run_id is passed to to constructor then all recording and logging will
    happen against that run_id.
    If no run_id is passed into constructor, then the run ID will be fetched
    from `mlflow.active_run()` each time `record_metrics()` or `flush()` is called; in this
    case, callers must ensure that an active run is present before invoking
    `record_metrics()` or `flush()`.
    """

    def __init__(self, run_id=None):
        self.run_id = run_id

        # data is an array of Metric objects
        self.data = []
        self.total_training_time = 0
        self.total_log_batch_time = 0
        self.previous_training_timestamp = None

    def flush(self):
        """
        The metrics accumulated by BatchMetricsLogger will be batch logged to an MLFlow run.
        """
        self._timed_log_batch()
        self.data = []

    def _timed_log_batch(self):
        if self.run_id is None:
            # Retrieving run_id from active mlflow run.
            current_run_id = mlflow.active_run().info.run_id
        else:
            current_run_id = self.run_id

        start = time.time()
        metrics_slices = [
            self.data[i : i + MAX_METRICS_PER_BATCH]
            for i in range(0, len(self.data), MAX_METRICS_PER_BATCH)
        ]
        for metrics_slice in metrics_slices:
            try_mlflow_log(MlflowClient().log_batch, run_id=current_run_id, metrics=metrics_slice)
        end = time.time()
        self.total_log_batch_time += end - start

    def _should_flush(self):
        target_training_to_logging_time_ratio = 10
        if (
            self.total_training_time
            >= self.total_log_batch_time * target_training_to_logging_time_ratio
        ):
            return True

        return False

    def record_metrics(self, metrics, step=None):
        """
        Submit a set of metrics to be logged. The metrics may not be immediately logged, as this
        class will batch them in order to not increase execution time too much by logging
        frequently.

        :param metrics: dictionary containing key, value pairs of metrics to be logged.
        :param step: the training step that the metrics correspond to.
        """
        current_timestamp = time.time()
        if self.previous_training_timestamp is None:
            self.previous_training_timestamp = current_timestamp

        training_time = current_timestamp - self.previous_training_timestamp

        self.total_training_time += training_time

        # log_batch() requires step to be defined. Therefore will set step to 0 if not defined.
        if step is None:
            step = 0

        for key, value in metrics.items():

            self.data.append(Metric(key, value, int(current_timestamp * 1000), step))

        if self._should_flush():
            self.flush()

        self.previous_training_timestamp = current_timestamp


@contextlib.contextmanager
def batch_metrics_logger(run_id):
    """
    Context manager that yields a BatchMetricsLogger object, which metrics can be logged against.
    The BatchMetricsLogger keeps metrics in a list until it decides they should be logged, at
    which point the accumulated metrics will be batch logged. The BatchMetricsLogger ensures
    that logging imposes no more than a 10% overhead on the training, where the training is
    measured by adding up the time elapsed between consecutive calls to record_metrics.

    If logging a batch fails, a warning will be emitted and subsequent metrics will continue to
    be collected.

    Once the context is closed, any metrics that have yet to be logged will be logged.

    :param run_id: ID of the run that the metrics will be logged to.
    """

    batch_metrics_logger = BatchMetricsLogger(run_id)
    yield batch_metrics_logger
    batch_metrics_logger.flush()


def _check_version_in_range(ver, min_ver, max_ver):
    return LooseVersion(min_ver) <= LooseVersion(ver) <= LooseVersion(max_ver)


def _load_version_file_as_dict():
    version_file_path = resource_filename(__name__, "../../ml-package-versions.yml")
    with open(version_file_path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


_module_version_info_dict = _load_version_file_as_dict()


# A map FLAVOR_NAME -> a tuple of (dependent_module_name, key_in_module_version_info_dict)
_cross_tested_flavor_to_module_name_and_module_key = {
    "fastai": ("fastai", "fastai-1.x"),
    "gluon": ("mxnet", "gluon"),
    "keras": ("keras", "keras"),
    "lightgbm": ("lightgbm", "lightgbm"),
    "statsmodels": ("statsmodels", "statsmodels"),
    "tensorflow": ("tensorflow", "tensorflow"),
    "xgboost": ("xgboost", "xgboost"),
    "sklearn": ("sklearn", "sklearn"),
    "pytorch": ("pytorch_lightning", "pytorch-lightning"),
}


def _get_min_max_version_and_pip_release(module_key):
    min_version = _module_version_info_dict[module_key]["autologging"]["minimum"]
    max_version = _module_version_info_dict[module_key]["autologging"]["maximum"]
    pip_release = _module_version_info_dict[module_key]["package_info"]["pip_release"]
    return min_version, max_version, pip_release


def _is_autologging_integration_supported(flavor_name):
    """
    :return: True if the flavor's associated package version is compatible with mlflow,
             False otherwise.
    """
    module_name, module_key = _cross_tested_flavor_to_module_name_and_module_key[flavor_name]
    actual_version = importlib.import_module(module_name).__version__
    min_version, max_version, _ = _get_min_max_version_and_pip_release(module_key)
    return _check_version_in_range(actual_version, min_version, max_version)


def _gen_autologging_package_version_requirements_doc(flavor_name):
    """
    :return: A document note string saying the compatibility for the flavor's associated package
             versions.
    """
    _, module_key = _cross_tested_flavor_to_module_name_and_module_key[flavor_name]
    min_ver, max_ver, pip_release = _get_min_max_version_and_pip_release(module_key)
    required_pkg_versions = "``{min_ver}`` <= ``{pip_release}`` <= ``{max_ver}``".format(
        min_ver=min_ver, pip_release=pip_release, max_ver=max_ver
    )

    return (
        "    .. Note:: Autologging is known to be compatible with the following package versions: "
        + required_pkg_versions
        + ". Autologging may not succeed when used with package versions outside of this range."
        + "\n\n"
    )


def _check_and_log_warning_for_unsupported_integration(flavor_name):
    """
    When autologging enabled disable_for_unsupported_versions disabled, check whether the flavor
    package version is compatible with mlflow, if not compatible, log a warning message.
    """
    if (
        flavor_name in _cross_tested_flavor_to_module_name_and_module_key
        and not get_autologging_config(flavor_name, "disable", True)
        and not get_autologging_config(flavor_name, "disable_for_unsupported_versions", False)
        and not _is_autologging_integration_supported(flavor_name)
    ):
        _logger.warning(
            "You are using an unsupported version of %s. If you encounter errors during "
            "autologging, try upgrading / downgrading %s to a supported version, or try "
            "upgrading MLflow.",
            flavor_name,
            flavor_name,
        )


def autologging_integration(name):
    """
    **All autologging integrations should be decorated with this wrapper.**

    Wraps an autologging function in order to store its configuration arguments. This enables
    patch functions to broadly obey certain configurations (e.g., disable=True) without
    requiring specific logic to be present in each autologging integration.
    """

    def validate_param_spec(param_spec):
        if "disable" not in param_spec or param_spec["disable"].default is not False:
            raise Exception(
                "Invalid `autolog()` function for integration '{}'. `autolog()` functions"
                " must specify a 'disable' argument with default value 'False'".format(name)
            )
        elif "silent" not in param_spec or param_spec["silent"].default is not False:
            raise Exception(
                "Invalid `autolog()` function for integration '{}'. `autolog()` functions"
                " must specify a 'silent' argument with default value 'False'".format(name)
            )

    def wrapper(_autolog):
        param_spec = inspect.signature(_autolog).parameters
        validate_param_spec(param_spec)

        AUTOLOGGING_INTEGRATIONS[name] = {}
        default_params = {param.name: param.default for param in param_spec.values()}

        def autolog(*args, **kwargs):
            config_to_store = dict(default_params)
            config_to_store.update(
                {param.name: arg for arg, param in zip(args, param_spec.values())}
            )
            config_to_store.update(kwargs)
            AUTOLOGGING_INTEGRATIONS[name] = config_to_store

            is_silent_mode = get_autologging_config(name, "silent", False)
            # Reroute non-MLflow warnings encountered during autologging enablement to an
            # MLflow event logger, and enforce silent mode if applicable (i.e. if the corresponding
            # autologging integration was called with `silent=True`)
            with set_mlflow_events_and_warnings_behavior_globally(
                # MLflow warnings emitted during autologging setup / enablement are likely
                # actionable and relevant to the user, so they should be emitted as normal
                # when `silent=False`. For reference, see recommended warning and event logging
                # behaviors from https://docs.python.org/3/howto/logging.html#when-to-use-logging
                reroute_warnings=False,
                disable_event_logs=is_silent_mode,
                disable_warnings=is_silent_mode,
            ), set_non_mlflow_warnings_behavior_for_current_thread(
                # non-MLflow warnings emitted during autologging setup / enablement are not
                # actionable for the user, as they are a byproduct of the autologging
                # implementation. Accordingly, they should be rerouted to `logger.warning()`.
                # For reference, see recommended warning and event logging
                # behaviors from https://docs.python.org/3/howto/logging.html#when-to-use-logging
                reroute_warnings=True,
                disable_warnings=is_silent_mode,
            ):

                try:
                    # Pass `autolog()` arguments to `log_autolog_called` in keyword format to enable
                    # event loggers to more easily identify important configuration parameters
                    # (e.g., `disable`) without examining positional arguments. Passing positional
                    # arguments to `log_autolog_called` is deprecated in MLflow > 1.13.1
                    AutologgingEventLogger.get_logger().log_autolog_called(
                        name, (), config_to_store
                    )
                except Exception:
                    pass

                _check_and_log_warning_for_unsupported_integration(name)

                return _autolog(*args, **kwargs)

        wrapped_autolog = update_wrapper_extended(autolog, _autolog)
        # Set the autologging integration name as a function attribute on the wrapped autologging
        # function, allowing the integration name to be extracted from the function. This is used
        # during the execution of import hooks for `mlflow.autolog()`.
        wrapped_autolog.integration_name = name

        if name in _cross_tested_flavor_to_module_name_and_module_key:
            wrapped_autolog.__doc__ = (
                _gen_autologging_package_version_requirements_doc(name) + wrapped_autolog.__doc__
            )
        return wrapped_autolog

    return wrapper


def get_autologging_config(flavor_name, config_key, default_value=None):
    """
    Returns a desired config value for a specified autologging integration.
    Returns `None` if specified `flavor_name` has no recorded configs.
    If `config_key` is not set on the config object, default value is returned.

    :param flavor_name: An autologging integration flavor name.
    :param config_key: The key for the desired config value.
    :param default_value: The default_value to return
    """
    config = AUTOLOGGING_INTEGRATIONS.get(flavor_name)
    if config is not None:
        return config.get(config_key, default_value)
    else:
        return default_value


def autologging_is_disabled(flavor_name):
    """
    Returns a boolean flag of whether the autologging integration is disabled.

    :param flavor_name: An autologging integration flavor name.
    """
    explicit_disabled = get_autologging_config(flavor_name, "disable", True)
    if explicit_disabled:
        return True

    if (
        flavor_name in _cross_tested_flavor_to_module_name_and_module_key
        and not _is_autologging_integration_supported(flavor_name)
    ):
        return get_autologging_config(flavor_name, "disable_for_unsupported_versions", False)

    return False
