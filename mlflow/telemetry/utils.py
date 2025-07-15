import inspect
import os
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY


# TODO: add check for CI  and mlflow dev versions,
# and other scenarios where telemetry should be disabled
def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get() or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
    )


def _get_whitelist() -> dict[str, set[str]]:
    """
    Whitelist for APIs that are only invoked by MLflow but should be tracked.
    """
    whitelist = defaultdict(set)
    try:
        from mlflow.pyfunc.utils.data_validation import _infer_schema_from_list_type_hint

        whitelist[_infer_schema_from_list_type_hint.__module__].add(
            _infer_schema_from_list_type_hint.__qualname__
        )
    except ImportError:
        pass

    return whitelist


def should_skip_telemetry(func) -> bool:
    # If the function is in whitelist, we should always track it
    if func.__qualname__ in _get_whitelist().get(func.__module__, set()):
        return False

    if _disable_telemetry_tracking_var.get():
        return True

    frame = inspect.currentframe()
    try:
        # skip the current frame and the API call frames
        frame = frame.f_back.f_back if frame and frame.f_back else None
        module = inspect.getmodule(frame)
        # TODO: consider recording if this comes from databricks modules
        return module and module.__name__.startswith("mlflow")
    finally:
        del frame


# ContextVar to disable telemetry tracking in the current thread.
# This is thread-local to avoid race conditions when multiple threads are running in parallel.
# NB: this doesn't work if a nested function spawns a new thread (e.g. mlflow.genai.evaluate)
_disable_telemetry_tracking_var = ContextVar("disable_telemetry_tracking", default=False)


@contextmanager
def _disable_telemetry():
    """
    Context manager to disable telemetry tracking in the following scenarios:
    1. Circular API calls: When MLflow invokes `databricks-agents` APIs, which in turn call back
        into MLflow APIs. This prevents telemetry from tracking internal, nested invocations.
    2. Code-based model logging: During model logging, the model file may be executed directly,
        potentially triggering additional telemetry logging inside model file. This context
        suppresses such telemetry during model loading and logging.
    """
    token = _disable_telemetry_tracking_var.set(True)
    try:
        yield
    finally:
        _disable_telemetry_tracking_var.reset(token)
