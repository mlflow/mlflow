import logging
from typing import TYPE_CHECKING, Union

from opentelemetry.sdk.trace import Span

from mlflow.tracing.utils import set_span_chat_tools
from mlflow.types.chat import ChatTool

if TYPE_CHECKING:
    from autogen_core.tools import BaseTool, ToolSchema

_logger = logging.getLogger(__name__)


def log_tools(span: Span, tools: list[Union["BaseTool", "ToolSchema"]]):
    """
    Log Autogen tool definitions into the passed in span.

    Ref: https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/tools.html

    Args:
        span: The span to log the tools into.
        tools: A list of Autogen BaseTool.
    """
    from autogen_core.tools import BaseTool

    try:
        tools = [
            ChatTool(
                type="function",
                function=tool.schema if isinstance(tool, BaseTool) else tool,
            )
            for tool in tools
        ]
        set_span_chat_tools(span, tools)
    except Exception:
        _logger.debug(f"Failed to log tools to Span {span}.", exc_info=True)
