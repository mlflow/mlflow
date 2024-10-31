from contextlib import contextmanager
from threading import Thread

from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
)
from mlflow.utils.databricks_utils import is_in_databricks_runtime

FLAVOR_NAME = "litellm"


@experimental
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
    _autolog(log_traces=log_traces, disable=disable, silent=silent)

    if log_traces and not disable:
        litellm.success_callback = _append_mlflow_callbacks(litellm.success_callback)
        litellm.failure_callback = _append_mlflow_callbacks(litellm.failure_callback)

        if is_in_databricks_runtime():
            # Patch main APIs e.g. completion to inject custom handling for threading.
            # By default, those API will start a new thread when calling log_success_event()
            # handler of the logging callbacks and never wait for it to finish. This is
            # problematic in Databricks notebook, because the inline trace UI display
            # assumes that the trace is generated synchronously. If the trace is generated
            # asynchronously, it will be displayed in different later cells.
            # To workaround this issue, we monkey-patch these APIs to wait for the logging
            # threads to finish before returning the result.
            # This is not required for OSS environment where we don't show inline trace UI.
            for func in [
                "completion",
                "embedding",
                "text_completion",
                "image_generation",
                "transcription",
                "speech",
            ]:
                _patch_sync_function(litellm, func)
                _patch_async_function(litellm, "a" + func)

            # For streaming case, we need to patch the iterator because traces are generated
            # when consuming the generator, not when calling the main APIs.
            _patch_sync_function(litellm.utils.CustomStreamWrapper, "__next__")
            _patch_async_function(litellm.utils.CustomStreamWrapper, "__anext__")
    else:
        litellm.success_callback = _remove_mlflow_callbacks(litellm.success_callback)
        litellm.failure_callback = _remove_mlflow_callbacks(litellm.failure_callback)
        # Callback also needs to be removed from 'callbacks' as litellm adds
        # success/failure callbacks to there as well.
        litellm.callbacks = _remove_mlflow_callbacks(litellm.callbacks)

        # Remove all patches applied to async functions (not via safe_patch())
        _unpatch()


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


def _patch_sync_function(target, function_name: str):
    """
    Apply the threading patch to a synchronous function.

    We capture the threads started by the function using the _patch_thread_start context manager,
    then join them to ensure they are finished before the notebook cell finishes executing.
    """

    def _patch_fn(original, *args, **kwargs):
        with _patch_thread_start() as logging_threads:
            result = original(*args, **kwargs)
        for thread in logging_threads:
            thread.join()
        return result

    safe_patch(FLAVOR_NAME, target, function_name, _patch_fn)


_ASYNC_PATCH_STORE = {}


def _patch_async_function(target, function_name: str):
    """Apply the threading patch to an async function."""
    if (target, function_name) in _ASYNC_PATCH_STORE:
        return

    original = getattr(target, function_name)

    async def _patch_fn(*args, **kwargs):
        with _patch_thread_start() as logging_threads:
            result = await original(*args, **kwargs)
        for thread in logging_threads:
            thread.join()
        return result

    # NB: safe_patch does not support async functions, so we simply use setattr method here.
    # To avoid double patching and handle cleanup, we store the patched function in a global dict.
    setattr(target, function_name, _patch_fn)
    _ASYNC_PATCH_STORE[(target, function_name)] = original


def _unpatch():
    """Remove the threading patch for all async functions."""
    for (target, function_name), original in _ASYNC_PATCH_STORE.items():
        setattr(target, function_name, original)
    _ASYNC_PATCH_STORE.clear()


@contextmanager
def _patch_thread_start():
    """
    A context manager to collect threads started for logging handlers.
    This is done by monkey-patching the start() method of threading.Thread.
    Note that failure handlers are executed synchronously, so we don't need to patch them.
    """
    original = Thread.start
    logging_threads = []

    def patched_thread(self, *args, **kwargs):
        target = getattr(self, "_target", None)
        # success_handler is for normal request, and run_success_... is for streaming
        # - https://github.com/BerriAI/litellm/blob/4f8a3fd4cfc20cf43b38379928b41c2691c85d36/litellm/utils.py#L946
        # - https://github.com/BerriAI/litellm/blob/4f8a3fd4cfc20cf43b38379928b41c2691c85d36/litellm/utils.py#L7526
        if target and target.__name__ in [
            "success_handler",
            "run_success_logging_and_cache_storage",
        ]:
            logging_threads.append(self)
        return original(self, *args, **kwargs)

    Thread.start = patched_thread
    try:
        yield logging_threads
    finally:
        Thread.start = original


def _append_mlflow_callbacks(original_callbacks):
    if not any(cb == "mlflow" for cb in original_callbacks):
        return original_callbacks + ["mlflow"]
    return original_callbacks


def _remove_mlflow_callbacks(original_callbacks):
    from litellm.integrations.mlflow import MlflowLogger

    return [cb for cb in original_callbacks if not (cb == "mlflow" or isinstance(cb, MlflowLogger))]
