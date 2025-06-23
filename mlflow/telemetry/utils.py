import os
import threading
from contextlib import contextmanager

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY

_mlflow_disable_telemetry_lock = threading.RLock()


@contextmanager
def temporarily_disable_telemetry():
    original_value = MLFLOW_DISABLE_TELEMETRY.get() if MLFLOW_DISABLE_TELEMETRY.is_set() else None
    with _mlflow_disable_telemetry_lock:
        try:
            MLFLOW_DISABLE_TELEMETRY.set(True)
            yield
        finally:
            if original_value is None:
                MLFLOW_DISABLE_TELEMETRY.unset()
            else:
                MLFLOW_DISABLE_TELEMETRY.set(original_value)


# TODO: add check for CI  and mlflow dev versions,
# and other scenarios where telemetry should be disabled
def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get() or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
    )
