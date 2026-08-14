import logging
from typing import Callable

from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "litellm"

_logger = logging.getLogger(__name__)


def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from LiteLLM to MLflow. Currently, MLflow
    only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for LiteLLM calls. If ``False``,
            no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the LiteLLM autologging integration. If ``False``,
            enables the LiteLLM autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during LiteLLM
            autologging. If ``False``, show all events and warnings.
    """
    import litellm

    # This needs to be called before doing any safe-patching (otherwise safe-patch will be no-op).
    # TODO: since this implementation is inconsistent, explore a universal way to solve the issue.
    _autolog(log_traces=log_traces, disable=disable, silent=silent)

    try:
        from litellm.integrations.mlflow import MlflowLogger  # noqa: F401
    except ImportError:
        _logger.warning(
            "MLflow LiteLLM integration is not supported for the installed LiteLLM version. "
            "Please upgrade to a newer version to enable MLflow LiteLLM autologging."
        )
        return

    if log_traces and not disable:
        litellm.success_callback = _append_mlflow_callbacks(litellm.success_callback)
        litellm.failure_callback = _append_mlflow_callbacks(litellm.failure_callback)

        # Patch thread pool executor to bypass non-blocking behavior of success_handler
        _patch_thread_pool()

    else:
        litellm.success_callback = _remove_mlflow_callbacks(litellm.success_callback)
        litellm.failure_callback = _remove_mlflow_callbacks(litellm.failure_callback)
        # Callback also needs to be removed from 'callbacks' as litellm adds
        # success/failure callbacks to there as well.
        litellm.callbacks = _remove_mlflow_callbacks(litellm.callbacks)

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


# NB: The @autologging_integration annotation must be applied here, and the callback injection
# needs to happen outside the annotated function. This is because the annotated function is NOT
# executed when disable=True is passed. This prevents us from removing our callback and patching
# when autologging is turned off.
@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool,
    disable: bool = False,
    silent: bool = False,
):
    pass


def _patch_thread_pool():
    """
    Apply the threading patch to a synchronous function.

    We capture the threads started by the function using the _patch_thread_start context manager,
    then join them to ensure they are finished before the notebook cell finishes executing.
    """
    try:
        from litellm.litellm_core_utils.thread_pool_executor import executor
    except ImportError:
        _logger.warning(
            "MLflow LiteLLM integration is not supported for the installed LiteLLM version. "
            "The behavior might be unstable."
        )
        return

    def _patched_submit(original, *args, **kwargs):
        # In litellm < 1.78, the success_handler is submitted directly.
        # In litellm >= 1.78, it's wrapped in a function named "run".
        fn_name = getattr(args[0], "__name__", "") if args else ""
        if args and isinstance(args[0], Callable) and fn_name in ("success_handler", "run"):
            # Immediately run the callback handler instead of submitting it to the thread pool
            args[0](*args[1:], **kwargs)
            return
        return original(*args, **kwargs)

    safe_patch(FLAVOR_NAME, executor, "submit", _patched_submit)


def _append_mlflow_callbacks(callbacks):
    from litellm.integrations.mlflow import MlflowLogger

    # MLflow callback can be stored as a string or the actual logger object
    if not any(cb == "mlflow" or isinstance(cb, MlflowLogger) for cb in callbacks):
        return callbacks + ["mlflow"]

    return callbacks


def _remove_mlflow_callbacks(callbacks):
    from litellm.integrations.mlflow import MlflowLogger

    return [cb for cb in callbacks if not (cb == "mlflow" or isinstance(cb, MlflowLogger))]
