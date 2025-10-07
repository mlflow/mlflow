"""
Tool registry for MLflow GenAI judges.

This module provides a registry system for managing and invoking JudgeTool instances.
"""

import json
import logging
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.4.0")
class JudgeToolRegistry:
    """Registry for managing and invoking JudgeTool instances."""

    def __init__(self):
        self._tools: dict[str, JudgeTool] = {}

    def register(self, tool: JudgeTool) -> None:
        """
        Register a judge tool in the registry.

        Args:
            tool: The JudgeTool instance to register
        """
        self._tools[tool.name] = tool

    def invoke(self, tool_call: Any, trace: Trace) -> Any:
        """
        Invoke a tool using a ToolCall instance and trace.

        Args:
            tool_call: The ToolCall containing function name and arguments
            trace: The MLflow trace object to analyze

        Returns:
            The result of the tool execution

        Raises:
            MlflowException: If the tool is not found or arguments are invalid
        """
        function_name = tool_call.function.name

        if function_name not in self._tools:
            raise MlflowException(
                f"Tool '{function_name}' not found in registry", error_code=RESOURCE_DOES_NOT_EXIST
            )
        tool = self._tools[function_name]

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Invalid JSON arguments for tool '{function_name}': {e}",
                error_code="INVALID_PARAMETER_VALUE",
            )

        _logger.debug(f"Invoking tool '{function_name}' with args: {arguments}")
        try:
            result = tool.invoke(trace, **arguments)
            _logger.debug(f"Tool '{function_name}' returned: {result}")
            return result
        except TypeError as e:
            raise MlflowException(
                f"Invalid arguments for tool '{function_name}': {e}",
                error_code="INVALID_PARAMETER_VALUE",
            )

    def list_tools(self) -> list[JudgeTool]:
        """
        List all registered tools.

        Returns:
            List of registered JudgeTool instances
        """
        return list(self._tools.values())


_judge_tool_registry = JudgeToolRegistry()


@experimental(version="3.4.0")
def register_judge_tool(tool: JudgeTool) -> None:
    """
    Register a judge tool in the global registry.

    Args:
        tool: The JudgeTool instance to register
    """
    _judge_tool_registry.register(tool)


@experimental(version="3.4.0")
def invoke_judge_tool(tool_call: Any, trace: Trace) -> Any:
    """
    Invoke a judge tool using a ToolCall instance and trace.

    Args:
        tool_call: The ToolCall containing function name and arguments
        trace: The MLflow trace object to analyze

    Returns:
        The result of the tool execution
    """
    return _judge_tool_registry.invoke(tool_call, trace)


@experimental(version="3.4.0")
def list_judge_tools() -> list[JudgeTool]:
    """
    List all registered judge tools.

    Returns:
        List of registered JudgeTool instances
    """
    return _judge_tool_registry.list_tools()


# NB: Tool imports are at the bottom to avoid circular dependencies and ensure
# the registry is fully defined before tools attempt to register themselves.
from mlflow.genai.judges.tools.get_root_span import GetRootSpanTool
from mlflow.genai.judges.tools.get_span import GetSpanTool
from mlflow.genai.judges.tools.get_span_performance_and_timing_report import (
    GetSpanPerformanceAndTimingReportTool,
)
from mlflow.genai.judges.tools.get_trace_info import GetTraceInfoTool
from mlflow.genai.judges.tools.list_spans import ListSpansTool
from mlflow.genai.judges.tools.search_trace_regex import SearchTraceRegexTool

_judge_tool_registry.register(GetTraceInfoTool())
_judge_tool_registry.register(GetRootSpanTool())
_judge_tool_registry.register(GetSpanTool())
_judge_tool_registry.register(ListSpansTool())
_judge_tool_registry.register(SearchTraceRegexTool())
_judge_tool_registry.register(GetSpanPerformanceAndTimingReportTool())
