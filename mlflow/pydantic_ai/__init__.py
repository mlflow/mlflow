import functools
import inspect
import logging
import typing

from mlflow.pydantic_ai.autolog import (
    patched_agent_init,
    patched_async_class_call,
    patched_async_stream_call,
    patched_capability_model_request,
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
        return_annotation = inspect.signature(func).return_annotation
    except (ValueError, TypeError):
        return False

    if return_annotation is inspect.Signature.empty:
        return False

    # pydantic-ai uses `from __future__ import annotations`, so the return
    # annotation is a raw string rather than a resolved type. We match by class
    # name to avoid calling `get_type_hints()`, which would try to resolve *all*
    # parameter annotations (e.g. `AgentSpec` added in 1.71.0) and raise
    # NameError for any forward reference that isn't importable at call time.
    # `StreamedRunResultSync` is a unique pydantic-ai class name; substring
    # matching is sufficient and avoids fragile import-time resolution.
    if isinstance(return_annotation, str):
        return "StreamedRunResultSync" in return_annotation

    origin = typing.get_origin(return_annotation) or return_annotation
    return hasattr(origin, "stream_text") and hasattr(origin, "stream_output")


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


def _get_tool_manager_module_path() -> str:
    """Return the importable module path for ToolManager.

    pydantic-ai >= 1.78.0 renamed _tool_manager to tool_manager (public).
    Try the public path first; fall back to the private one for older versions.
    """
    try:
        import pydantic_ai.tool_manager  # noqa: F401

        return "pydantic_ai.tool_manager"
    except ImportError:
        return "pydantic_ai._tool_manager"


def _tool_manager_uses_execute_tool_call() -> bool:
    """Return True if ToolManager has execute_tool_call (pydantic-ai >= 1.63.0).

    In pydantic-ai >= 1.63.0, _agent_graph._call_tool() calls
    tool_manager.execute_tool_call() directly rather than handle_call(), so we
    must patch execute_tool_call to capture the TOOL span.
    """
    module_path = _get_tool_manager_module_path()
    try:
        module = __import__(module_path, fromlist=["ToolManager"])
        return hasattr(module.ToolManager, "execute_tool_call")
    except ImportError:
        return False


def _has_instrumentation_capability() -> bool:
    """Return True if pydantic-ai routes model calls through the Instrumentation capability.

    pydantic-ai >= 1.95 replaced the ``InstrumentedModel`` wrapper with a capabilities
    system: model requests funnel through ``Instrumentation.wrap_model_request`` instead of
    ``InstrumentedModel.request``/``request_stream``. When this module is importable we
    instrument that hook and skip the ``InstrumentedModel`` patch, so exactly one path
    produces the LLM span (never both).
    """
    try:
        import pydantic_ai.capabilities.instrumentation  # noqa: F401

        return True
    except ImportError:
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
    # Base methods that exist in all supported versions
    agent_methods = ["run", "run_sync", "run_stream"]

    try:
        from pydantic_ai import Agent

        # run_stream_sync was added in pydantic-ai 1.10.0
        if hasattr(Agent, "run_stream_sync"):
            agent_methods.append("run_stream_sync")
    except ImportError:
        pass

    # On pydantic-ai >= 1.95 the Instrumentation capability (patched below) is the model
    # request funnel; on older versions it's InstrumentedModel. Instrument exactly one of
    # them so a single LLM span is produced per model call, never two overlapping spans.
    has_instrumentation_capability = _has_instrumentation_capability()

    tool_manager_path = f"{_get_tool_manager_module_path()}.ToolManager"
    class_map = {
        "pydantic_ai.Agent": agent_methods,
        # In pydantic-ai >= 1.63.0, _agent_graph calls execute_tool_call directly,
        # bypassing handle_call. Patch execute_tool_call when available; fall back to
        # handle_call for older versions where execute_tool_call doesn't exist.
        tool_manager_path: ["execute_tool_call"]
        if _tool_manager_uses_execute_tool_call()
        else ["handle_call"],
        "pydantic_ai.mcp.MCPServer": ["call_tool", "list_tools"],
    }
    if not has_instrumentation_capability:
        class_map["pydantic_ai.models.instrumented.InstrumentedModel"] = [
            "request",
            "request_stream",
        ]

    try:
        from pydantic_ai import Tool

        # Tool.run method is removed in recent versions
        if hasattr(Tool, "run"):
            class_map["pydantic_ai.Tool"] = ["run"]
    except ImportError:
        pass

    # Patch Agent.__init__ to auto-enable instrument=True so LLM spans
    # are captured without requiring users to explicitly set it
    try:
        from pydantic_ai import Agent

        original_init = Agent.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            return patched_agent_init(original_init, self, *args, **kwargs)

        patch = _wrap_patch(Agent, "__init__", patched_init)
        _store_patch(FLAVOR_NAME, patch)
    except (ImportError, AttributeError) as e:
        _logger.error("Error patching Agent.__init__: %s", e)

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

    # pydantic-ai >= 1.95 routes both streaming and non-streaming model requests through
    # the Instrumentation capability's wrap_model_request instead of InstrumentedModel.
    # Patch it (and only it, since InstrumentedModel is excluded from class_map above) so a
    # single provider-agnostic LLM span is produced per model request on modern versions.
    if has_instrumentation_capability:
        try:
            from pydantic_ai.capabilities.instrumentation import Instrumentation

            safe_patch(
                FLAVOR_NAME,
                Instrumentation,
                "wrap_model_request",
                patched_capability_model_request,
            )
        except (ImportError, AttributeError) as e:
            _logger.error("Error patching Instrumentation.wrap_model_request: %s", e)

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
