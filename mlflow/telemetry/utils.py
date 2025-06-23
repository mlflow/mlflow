import inspect
import os

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY


# TODO: add check for CI  and mlflow dev versions,
# and other scenarios where telemetry should be disabled
def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get() or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
    )


def invoked_from_internal_api() -> bool:
    frame = inspect.currentframe()
    try:
        # skip the current frame and the API call frame
        frame = frame.f_back.f_back if frame and frame.f_back else None
        module = inspect.getmodule(frame)
        return module and module.__name__.startswith("mlflow")
    finally:
        del frame
