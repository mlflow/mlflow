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
def _invoked_from_internal_api() -> bool:
    frame = inspect.currentframe()
    try:
        # skip the current frame and the API call frames
        # current frame: mlflow.telemetry.utils -_invoked_from_internal_api
        # last frame: mlflow.telemetry.utils - is_invoked_from_internal_api
        # second last frame: mlflow.telemetry.track - track_api_usage
        frame = (
            frame.f_back.f_back.f_back if frame and frame.f_back and frame.f_back.f_back else None
        )
        module = inspect.getmodule(frame)
        return module and module.__name__.startswith("mlflow")
    finally:
        del frame


# ContextVar to track if the current function is invoked from an internal API
# This is thread-local to avoid race conditions when multiple threads are running in parallel.
_invoked_from_internal_api_var = ContextVar(
    "invoked_from_internal_api", default=_invoked_from_internal_api
)


def is_invoked_from_internal_api() -> bool:
    """
    Check if the current function is invoked from another MLflow API.
    """
    return _invoked_from_internal_api_var.get()()


@contextmanager
def _avoid_telemetry_tracking():
    """
    Context manager to disable telemetry tracking in the following scenarios:
    1. Circular API calls: When MLflow invokes Databricks Agents APIs, which in turn call back
        into MLflow APIs. This prevents telemetry from tracking internal, nested invocations.
    2. Code-based model logging: During model logging, the model file may be executed directly,
        potentially triggering additional telemetry logging inside model file. This context
        suppresses such telemetry during model loading and logging.
    """

    def mock_invoked_from_internal_api():
        return True

    token = _invoked_from_internal_api_var.set(mock_invoked_from_internal_api)
    try:
        yield
    finally:
        _invoked_from_internal_api_var.reset(token)
