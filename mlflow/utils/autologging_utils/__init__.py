# pylint: disable=unused-wildcard-import,wildcard-import

import inspect
import logging
import time
import contextlib

import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH

# Define the module-level logger for autologging utilities before importing utilities defined in
# submodules (e.g., `safety`, `events`) that depend on the module-level logger. Add the `noqa: E402`
# comment after each subsequent import to ignore "import not at top of file" code style errors
_logger = logging.getLogger(__name__)

# Import autologging utilities used by this module
from mlflow.utils.autologging_utils.logging_and_warnings import (
    set_mlflow_events_and_warnings_behavior_globally,
    set_non_mlflow_warnings_behavior_for_current_thread,
)
from mlflow.utils.autologging_utils.safety import (
    update_wrapper_extended,
    revert_patches,
)
from mlflow.utils.autologging_utils.versioning import (
    FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY,
    get_min_max_version_and_pip_release,
    is_flavor_supported_for_associated_package_versions,
)

# Wildcard import other autologging utilities (e.g. safety utilities, event logging utilities) used
# in autologging integration implementations, which reference them via the
# `mlflow.utils.autologging_utils` module
from mlflow.utils.autologging_utils.safety import *
from mlflow.utils.autologging_utils.events import *
from mlflow.utils.autologging_utils.client import *


INPUT_EXAMPLE_SAMPLE_ROWS = 5
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)
_AUTOLOGGING_TEST_MODE_ENV_VAR = "MLFLOW_AUTOLOGGING_TESTING"

# Flag indicating whether autologging is globally disabled for all integrations.
_AUTOLOGGING_GLOBALLY_DISABLED = False

# Autologging config key indicating whether or not a particular autologging integration
# was configured (i.e. its various `log_models`, `disable`, etc. configuration options
# were set) via a call to `mlflow.autolog()`, rather than via a call to the integration-specific
# autologging method (e.g., `mlflow.tensorflow.autolog()`, ...)
AUTOLOGGING_CONF_KEY_IS_GLOBALLY_CONFIGURED = "globally_configured"

# Dict mapping integration name to its config.
AUTOLOGGING_INTEGRATIONS = {}

_logger = logging.getLogger(__name__)


def get_mlflow_run_params_for_fn_args(fn, args, kwargs, unlogged=None):
    """
    Given arguments explicitly passed to a function, generate a dictionary of MLflow Run
    parameter key / value pairs.

    :param fn: function whose parameters are to be logged
    :param args: arguments explicitly passed into fn. If `fn` is defined on a class,
                 `self` should not be part of `args`; the caller is responsible for
                 filtering out `self` before calling this function.
    :param kwargs: kwargs explicitly passed into fn
    :param unlogged: parameters not to be logged
    :return: A dictionary of MLflow Run parameter key / value pairs.
    """
    unlogged = unlogged or []
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
    return params_to_log


def log_fn_args_as_params(fn, args, kwargs, unlogged=None):  # pylint: disable=W0102
    """
    Log arguments explicitly passed to a function as MLflow Run parameters to the current active
    MLflow Run.

    :param fn: function whose parameters are to be logged
    :param args: arguments explicitly passed into fn. If `fn` is defined on a class,
                 `self` should not be part of `args`; the caller is responsible for
                 filtering out `self` before calling this function.
    :param kwargs: kwargs explicitly passed into fn
    :param unlogged: parameters not to be logged
    :return: None
    """
    params_to_log = get_mlflow_run_params_for_fn_args(fn, args, kwargs, unlogged)
    mlflow.log_params(params_to_log)


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

    def __init__(self, run_id=None, tracking_uri=None):
        self.run_id = run_id
        self.client = MlflowClient(tracking_uri)

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
            self.client.log_batch(run_id=current_run_id, metrics=metrics_slice)
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


def gen_autologging_package_version_requirements_doc(integration_name):
    """
    :return: A document note string saying the compatibility for the specified autologging
             integration's associated package versions.
    """
    _, module_key = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[integration_name]
    min_ver, max_ver, pip_release = get_min_max_version_and_pip_release(module_key)
    required_pkg_versions = f"``{min_ver}`` <= ``{pip_release}`` <= ``{max_ver}``"

    return (
        "    .. Note:: Autologging is known to be compatible with the following package versions: "
        + required_pkg_versions
        + ". Autologging may not succeed when used with package versions outside of this range."
        + "\n\n"
    )


def _check_and_log_warning_for_unsupported_package_versions(integration_name):
    """
    When autologging is enabled and `disable_for_unsupported_versions=False` for the specified
    autologging integration, check whether the currently-installed versions of the integration's
    associated package versions are supported by the specified integration. If the package versions
    are not supported, log a warning message.
    """
    if (
        integration_name in FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY
        and not get_autologging_config(integration_name, "disable", True)
        and not get_autologging_config(integration_name, "disable_for_unsupported_versions", False)
        and not is_flavor_supported_for_associated_package_versions(integration_name)
    ):
        _logger.warning(
            "You are using an unsupported version of %s. If you encounter errors during "
            "autologging, try upgrading / downgrading %s to a supported version, or try "
            "upgrading MLflow.",
            integration_name,
            integration_name,
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
                f"Invalid `autolog()` function for integration '{name}'. `autolog()` functions"
                " must specify a 'disable' argument with default value 'False'"
            )
        elif "silent" not in param_spec or param_spec["silent"].default is not False:
            raise Exception(
                f"Invalid `autolog()` function for integration '{name}'. `autolog()` functions"
                " must specify a 'silent' argument with default value 'False'"
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

            try:
                # Pass `autolog()` arguments to `log_autolog_called` in keyword format to enable
                # event loggers to more easily identify important configuration parameters
                # (e.g., `disable`) without examining positional arguments. Passing positional
                # arguments to `log_autolog_called` is deprecated in MLflow > 1.13.1
                AutologgingEventLogger.get_logger().log_autolog_called(name, (), config_to_store)
            except Exception:
                pass

            revert_patches(name)

            # If disabling autologging using fluent api, then every active integration's autolog
            # needs to be called with disable=True. So do not short circuit and let
            # `mlflow.autolog()` invoke all active integrations with disable=True.
            if name != "mlflow" and get_autologging_config(name, "disable", True):
                return

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
                _check_and_log_warning_for_unsupported_package_versions(name)

                return _autolog(*args, **kwargs)

        wrapped_autolog = update_wrapper_extended(autolog, _autolog)
        # Set the autologging integration name as a function attribute on the wrapped autologging
        # function, allowing the integration name to be extracted from the function. This is used
        # during the execution of import hooks for `mlflow.autolog()`.
        wrapped_autolog.integration_name = name

        if name in FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY:
            wrapped_autolog.__doc__ = gen_autologging_package_version_requirements_doc(name) + (
                wrapped_autolog.__doc__ or ""
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


def autologging_is_disabled(integration_name):
    """
    Returns a boolean flag of whether the autologging integration is disabled.

    :param integration_name: An autologging integration flavor name.
    """
    explicit_disabled = get_autologging_config(integration_name, "disable", True)
    if explicit_disabled:
        return True

    if (
        integration_name in FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY
        and not is_flavor_supported_for_associated_package_versions(integration_name)
    ):
        return get_autologging_config(integration_name, "disable_for_unsupported_versions", False)

    return False


@contextlib.contextmanager
def disable_autologging():
    """
    Context manager that temporarily disables autologging globally for all integrations upon
    entry and restores the previous autologging configuration upon exit.
    """
    global _AUTOLOGGING_GLOBALLY_DISABLED
    _AUTOLOGGING_GLOBALLY_DISABLED = True
    yield None
    _AUTOLOGGING_GLOBALLY_DISABLED = False


def _get_new_training_session_class():
    """
    Returns a session manager class for nested autologging runs.

    Examples
    --------
    >>> class Parent: pass
    >>> class Child: pass
    >>> class Grandchild: pass
    >>>
    >>> _TrainingSession = _get_new_training_session_class()
    >>> with _TrainingSession(Parent, False) as p:
    ...     with _SklearnTrainingSession(Child, True) as c:
    ...         with _SklearnTrainingSession(Grandchild, True) as g:
    ...             print(p.should_log(), c.should_log(), g.should_log())
    True False False
    >>>
    >>> with _TrainingSession(Parent, True) as p:
    ...     with _TrainingSession(Child, False) as c:
    ...         with _TrainingSession(Grandchild, True) as g:
    ...             print(p.should_log(), c.should_log(), g.should_log())
    True True False
    >>>
    >>> with _TrainingSession(Child, True) as c1:
    ...     with _TrainingSession(Child, True) as c2:
    ...             print(c1.should_log(), c2.should_log())
    True False
    """

    # NOTE: The current implementation doesn't guarantee thread-safety, but that's okay for now
    # because:
    # 1. We don't currently have any use cases for allow_children=True.
    # 2. The list append & pop operations are thread-safe, so we will always clear the session stack
    #    once all _TrainingSessions exit.
    class _TrainingSession:
        _session_stack = []

        def __init__(self, estimator, allow_children=True):
            """
            A session manager for nested autologging runs.

            :param estimator: An estimator that this session originates from.
            :param allow_children: If True, allows autologging in child sessions.
                                   If False, disallows autologging in all descendant sessions.
            """
            self.allow_children = allow_children
            self.estimator = estimator
            self._parent = None

        def __enter__(self):
            if len(_TrainingSession._session_stack) > 0:
                self._parent = _TrainingSession._session_stack[-1]
                self.allow_children = (
                    _TrainingSession._session_stack[-1].allow_children and self.allow_children
                )
            _TrainingSession._session_stack.append(self)
            return self

        def __exit__(self, tp, val, traceback):
            _TrainingSession._session_stack.pop()

        def should_log(self):
            """
            Returns True when at least one of the following conditions satisfies:

            1. This session is the root session.
            2. The parent session allows autologging and its estimator differs from this session's
               estimator.
            """
            for training_session in _TrainingSession._session_stack:
                if training_session is self:
                    break
                elif training_session.estimator is self.estimator:
                    return False

            return self._parent is None or self._parent.allow_children

        @staticmethod
        def is_active():
            return len(_TrainingSession._session_stack) != 0

        @staticmethod
        def get_current_session():
            if _TrainingSession.is_active():
                return _TrainingSession._session_stack[-1]
            return None

    return _TrainingSession


def get_instance_method_first_arg_value(method, call_pos_args, call_kwargs):
    """
    Get instance method first argument value (exclude the `self` argument).
    :param method A `cls.method` object which includes the `self` argument.
    :param call_pos_args: positional arguments excluding the first `self` argument.
    :param call_kwargs: keywords arguments.
    """
    if len(call_pos_args) >= 1:
        return call_pos_args[0]
    else:
        param_sig = inspect.signature(method).parameters
        first_arg_name = list(param_sig.keys())[1]
        assert param_sig[first_arg_name].kind not in [
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ]
        return call_kwargs.get(first_arg_name)


def get_method_call_arg_value(arg_index, arg_name, default_value, call_pos_args, call_kwargs):
    """
    Get argument value for a method call.
    :param arg_index: the argument index in the function signature. start from 0.
    :param arg_name: the argument name in the function signature.
    :param default_value: default argument value.
    :param call_pos_args: the positional argument values in the method call.
    :param call_kwargs: the keyword argument values in the method call.
    """
    if arg_name in call_kwargs:
        return call_kwargs[arg_name]
    elif arg_index < len(call_pos_args):
        return call_pos_args[arg_index]
    else:
        return default_value
