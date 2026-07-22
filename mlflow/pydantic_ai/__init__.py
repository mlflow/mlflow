import importlib.metadata

from packaging.version import Version

from mlflow.pydantic_ai.autolog import _get_tool_manager_module_path as _legacy_tool_manager_path
from mlflow.pydantic_ai.autolog import (
    _has_instrumentation_capability as _legacy_has_instrumentation_capability,
)
from mlflow.pydantic_ai.autolog import setup_autologging
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "pydantic_ai"


def _get_tool_manager_module_path() -> str:
    return _legacy_tool_manager_path()


def _tool_manager_uses_execute_tool_call() -> bool:
    module_path = _get_tool_manager_module_path()
    try:
        module = __import__(module_path, fromlist=["ToolManager"])
        return hasattr(module.ToolManager, "execute_tool_call")
    except ImportError:
        return False


def _has_instrumentation_capability() -> bool:
    return _legacy_has_instrumentation_capability()


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
        setup_autologging(
            get_tool_manager_module_path=_get_tool_manager_module_path,
            tool_manager_uses_execute_tool_call=_tool_manager_uses_execute_tool_call,
        )

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
