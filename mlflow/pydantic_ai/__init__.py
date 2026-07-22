import importlib.metadata

from packaging.version import Version

from mlflow.pydantic_ai.autolog import (
    _get_tool_manager_module_path,
    _has_instrumentation_capability,
    _tool_manager_uses_execute_tool_call,
    setup_autologging,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "pydantic_ai"


def _is_pydantic_ai_v2() -> bool:
    try:
        return Version(importlib.metadata.version("pydantic-ai")).major >= 2
    except importlib.metadata.PackageNotFoundError:
        return False


@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enable (or disable) autologging for Pydantic_AI.

    Args:
        log_traces: If True, capture spans for agent + model calls.
        disable:   If True, disable the autologging patches.
        silent:    If True, suppress MLflow warnings/info.
    """
    if _is_pydantic_ai_v2():
        from mlflow.pydantic_ai.autolog_v2 import setup_autologging as setup_v2_autologging

        setup_v2_autologging()
    else:
        setup_autologging()

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
