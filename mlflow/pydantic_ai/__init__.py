import inspect
import logging

from mlflow.pydantic_ai.autolog import (
    patched_async_class_call,
    patched_class_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "pydantic_ai"
_logger = logging.getLogger(__name__)


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enable (or disable) autologging for Pydantic_AI.

    Args:
        log_traces: If True, capture spans for agent + model calls.
        disable:   If True, disable the autologging patches.
        silent:    If True, suppress MLflow warnings/info.
    """
    class_map = {
        "pydantic_ai.Agent": ["run", "run_sync"],
        "pydantic_ai.models.instrumented.InstrumentedModel": ["request"],
        "pydantic_ai.Tool": ["run"],
        "pydantic_ai.mcp.MCPServer": ["call_tool", "list_tools"],
    }

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
                orig = getattr(cls, method)
                wrapper = (
                    patched_async_class_call
                    if inspect.iscoroutinefunction(orig)
                    else patched_class_call
                )
                safe_patch(
                    FLAVOR_NAME,
                    cls,
                    method,
                    wrapper,
                )
            except AttributeError as e:
                _logger.error("Error patching %s.%s: %s", cls_path, method, e)
