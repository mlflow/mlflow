import functools
import inspect
import logging
import typing

from mlflow.pydantic_ai.autolog import (
    patched_async_class_call,
    patched_async_stream_call,
    patched_class_call,
    patched_sync_stream_call,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.autologging_utils.safety import _store_patch, _wrap_patch

FLAVOR_NAME = "pydantic_ai"
_logger = logging.getLogger(__name__)


def _is_async_context_manager_factory(func) -> bool:
    wrapped = getattr(func, "__wrapped__", None)
    return wrapped is not None and inspect.isasyncgenfunction(wrapped)


def _returns_sync_streamed_result(func) -> bool:
    if inspect.iscoroutinefunction(func):
        return False

    try:
        hints = typing.get_type_hints(func)
        return_type = hints.get("return")
        if return_type is None:
            return False

        origin = typing.get_origin(return_type) or return_type

        return hasattr(origin, "stream_text") and hasattr(origin, "stream_output")
    except Exception:
        return False


def _patch_streaming_method(cls, method_name, wrapper_func):
    original = getattr(cls, method_name)

    @functools.wraps(original)
    def patched_method(self, *args, **kwargs):
        return wrapper_func(original, self, *args, **kwargs)

    patch = _wrap_patch(cls, method_name, patched_method)
    _store_patch(FLAVOR_NAME, patch)


def _patch_method(cls, method_name):
    method = getattr(cls, method_name)

    if _is_async_context_manager_factory(method):
        _patch_streaming_method(cls, method_name, patched_async_stream_call)
    elif _returns_sync_streamed_result(method):
        _patch_streaming_method(cls, method_name, patched_sync_stream_call)
    elif inspect.iscoroutinefunction(method):
        safe_patch(FLAVOR_NAME, cls, method_name, patched_async_class_call)
    else:
        safe_patch(FLAVOR_NAME, cls, method_name, patched_class_call)


@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enable (or disable) autologging for Pydantic_AI.

    Args:
        log_traces: If True, capture spans for agent + model calls.
        disable:   If True, disable the autologging patches.
        silent:    If True, suppress MLflow warnings/info.
    """
    # Base methods that exist in all supported versions
    agent_methods = ["run", "run_sync", "run_stream"]

    try:
        from pydantic_ai import Agent

        # run_stream_sync was added in pydantic-ai 1.10.0
        if hasattr(Agent, "run_stream_sync"):
            agent_methods.append("run_stream_sync")
    except ImportError:
        pass

    class_map = {
        "pydantic_ai.Agent": agent_methods,
        "pydantic_ai.models.instrumented.InstrumentedModel": [
            "request",
            "request_stream",
        ],
        "pydantic_ai._tool_manager.ToolManager": ["handle_call"],
        "pydantic_ai.mcp.MCPServer": ["call_tool", "list_tools"],
    }

    try:
        from pydantic_ai import Tool

        # Tool.run method is removed in recent versions
        if hasattr(Tool, "run"):
            class_map["pydantic_ai.Tool"] = ["run"]
    except ImportError:
        pass

    for cls_path, methods in class_map.items():
        module_name, class_name = cls_path.rsplit(".", 1)
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            _logger.error("Error importing %s: %s", cls_path, e)
            continue

        for method in methods:
            try:
                _patch_method(cls, method)
            except AttributeError as e:
                _logger.error("Error patching %s.%s: %s", cls_path, method, e)

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
