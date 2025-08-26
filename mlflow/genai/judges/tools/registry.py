"""
Tool registry for MLflow GenAI judges.

This module provides a registry system for managing and invoking JudgeTool instances.
"""

import json
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.types.llm import ToolCall
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class JudgeToolStore:
    """Judge tool storage implementation."""

    def __init__(self):
        self._tools: dict[str, JudgeTool] = {}

    def register_tool(self, tool: JudgeTool) -> None:
        """Register a judge tool in the store."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> JudgeTool:
        """Get a judge tool by name."""
        if name not in self._tools:
            raise MlflowException(
                f"Tool '{name}' not found in registry", error_code=RESOURCE_DOES_NOT_EXIST
            )
        return self._tools[name]

    def list_tools(self) -> list[JudgeTool]:
        """List all registered tools."""
        return list(self._tools.values())


@experimental(version="3.4.0")
class JudgeToolRegistry:
    """Registry for managing and invoking JudgeTool instances."""

    def __init__(self, store: JudgeToolStore = None):
        self._store = store or JudgeToolStore()

    def register(self, tool: JudgeTool) -> None:
        """
        Register a judge tool in the registry.

        Args:
            tool: The JudgeTool instance to register
        """
        self._store.register_tool(tool)

    def invoke(self, tool_call: ToolCall, trace: Trace) -> Any:
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

        # Get the tool from the registry
        tool = self._store.get_tool(function_name)

        # Parse the JSON arguments
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Invalid JSON arguments for tool '{function_name}': {e}",
                error_code="INVALID_PARAMETER_VALUE",
            )

        # Invoke the tool with the trace and parsed arguments
        try:
            return tool.invoke(trace, **arguments)
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
        return self._store.list_tools()


# Global registry instance
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
def invoke_judge_tool(tool_call: ToolCall, trace: Trace) -> Any:
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


# Register built-in tools - import at the end to avoid circular imports
def _register_builtin_tools():
    from mlflow.genai.judges.tools.get_trace_info import GetTraceInfoTool

    _judge_tool_registry.register(GetTraceInfoTool())


_register_builtin_tools()
