import inspect
import os
from contextlib import contextmanager
from contextvars import ContextVar

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY


# TODO: add check for CI  and mlflow dev versions,
# and other scenarios where telemetry should be disabled
def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get() or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
    )


# TODO: add whitelist for APIs that's only invoked by MLflow, e.g. MlflowV2SpanExporter.export
def invoked_from_internal_api() -> bool:
    frame = inspect.currentframe()
    try:
        # skip the current frame and the API call frames
        frame = frame.f_back.f_back if frame and frame.f_back else None
        module = inspect.getmodule(frame)
        # TODO: consider recording module name if False
        return module and module.__name__.startswith("mlflow")
    finally:
        del frame


# ContextVar to disable telemetry tracking in the current thread.
# This is thread-local to avoid race conditions when multiple threads are running in parallel.
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
