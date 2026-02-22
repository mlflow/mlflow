import contextlib
import io
import os
from typing import TYPE_CHECKING, Any, Callable

import click
from click.types import BOOL, FLOAT, INT, STRING, UUID

import mlflow.deployments.cli as deployments_cli
import mlflow.experiments
import mlflow.models.cli as models_cli
import mlflow.runs
from mlflow.ai_commands.ai_command_utils import get_command_body, list_commands
from mlflow.cli.scorers import commands as scorers_cli
from mlflow.cli.traces import commands as traces_cli
from mlflow.mcp.decorator import get_mcp_tool_name

# Environment variable to control which tool categories are enabled
# Supported values:
#   - "genai": traces, scorers, experiments, and runs tools (default)
#   - "ml": experiments, runs, models and deployments tools
#   - "all": all available tools
#   - Comma-separated list: "traces,scorers,experiments,runs,models,deployments"
MLFLOW_MCP_TOOLS = os.environ.get("MLFLOW_MCP_TOOLS", "genai")

# Tool category mappings
_GENAI_TOOLS = {"traces", "scorers", "experiments", "runs"}
_ML_TOOLS = {"models", "deployments", "experiments", "runs"}
_ALL_TOOLS = _GENAI_TOOLS | _ML_TOOLS

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.tools import FunctionTool


def param_type_to_json_schema_type(pt: click.ParamType) -> str:
    """
    Converts a Click ParamType to a JSON schema type.
    """
    if pt is STRING:
        return "string"
    if pt is BOOL:
        return "boolean"
    if pt is INT:
        return "integer"
    if pt is FLOAT:
        return "number"
    if pt is UUID:
        return "string"
    return "string"


def get_input_schema(params: list[click.Parameter]) -> dict[str, Any]:
    """
    Converts click params to JSON schema
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        schema = {
            "type": param_type_to_json_schema_type(p.type),
        }
        if p.default is not None and (
            # In click >= 8.3.0, the default value is set to `Sentinel.UNSET` when no default is
            # provided. Skip setting the default in this case.
            # See https://github.com/pallets/click/pull/3030 for more details.
            not isinstance(p.default, str) and repr(p.default) != "Sentinel.UNSET"
        ):
            schema["default"] = p.default
        if isinstance(p, click.Option):
            schema["description"] = (p.help or "").strip()
        if isinstance(p.type, click.Choice):
            schema["enum"] = [str(choice) for choice in p.type.choices]
        if p.required:
            required.append(p.name)
        properties[p.name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def fn_wrapper(command: click.Command) -> Callable[..., str]:
    def wrapper(**kwargs: Any) -> str:
        click_unset = getattr(click.core, "UNSET", object())

        # Capture stdout and stderr
        string_io = io.StringIO()
        with (
            contextlib.redirect_stdout(string_io),
            contextlib.redirect_stderr(string_io),
        ):
            # Fill in defaults for missing arguments
            # For Click 8.3.0+, we need to pass ALL parameters to the callback,
            # even those with Sentinel.UNSET defaults, so Click can handle them properly
            for param in command.params:
                if param.name not in kwargs:
                    kwargs[param.name] = param.default
            command.callback(**kwargs)  # type: ignore[misc]
        return string_io.getvalue().strip()

    return wrapper


def cmd_to_function_tool(cmd: click.Command) -> "FunctionTool | None":
    """
    Converts a Click command to a FunctionTool.

    Args:
        cmd: The Click command to convert.

    Returns:
        FunctionTool if the command has been decorated with @mlflow_mcp,
        None if the command should be skipped (not decorated for MCP exposure).
    """
    from fastmcp.tools import FunctionTool

    # Get the MCP tool name from the decorator
    tool_name = get_mcp_tool_name(cmd)

    # Skip commands that don't have the @mlflow_mcp decorator
    # This allows us to curate which commands are exposed as MCP tools
    if tool_name is None:
        return None

    return FunctionTool(
        fn=fn_wrapper(cmd),
        name=tool_name,
        description=(cmd.help or "").strip(),
        parameters=get_input_schema(cmd.params),
    )


def register_prompts(mcp: "FastMCP") -> None:
    """Register AI commands as MCP prompts."""
    from mlflow.telemetry.events import AiCommandRunEvent
    from mlflow.telemetry.track import _record_event

    for command in list_commands():
        # Convert slash-separated keys to underscores for MCP names
        mcp_name = command["key"].replace("/", "_")

        # Create a closure to capture the command key
        def make_prompt(cmd_key: str):
            @mcp.prompt(name=mcp_name, description=command["description"])
            def ai_command_prompt() -> str:
                """Execute an MLflow AI command prompt."""
                _record_event(AiCommandRunEvent, {"command_key": cmd_key, "context": "mcp"})
                return get_command_body(cmd_key)

            return ai_command_prompt

        # Register the prompt
        make_prompt(command["key"])


def _is_tool_enabled(category: str) -> bool:
    """Check if a tool category is enabled based on MLFLOW_MCP_TOOLS env var."""
    tools_config = MLFLOW_MCP_TOOLS.lower().strip()

    # Handle preset categories
    if tools_config == "all":
        return True
    if tools_config == "genai":
        return category.lower() in _GENAI_TOOLS
    if tools_config == "ml":
        return category.lower() in _ML_TOOLS

    # Handle comma-separated list of individual tools
    enabled_tools = {t.strip().lower() for t in tools_config.split(",")}
    return category.lower() in enabled_tools


def _collect_tools(commands: dict[str, click.Command]) -> list["FunctionTool"]:
    """Collect MCP tools from commands, filtering out undecorated commands."""
    tools = []
    for cmd in commands.values():
        tool = cmd_to_function_tool(cmd)
        if tool is not None:
            tools.append(tool)
    return tools


def create_mcp() -> "FastMCP":
    from fastmcp import FastMCP

    tools: list["FunctionTool"] = []

    # Traces CLI tools (genai)
    if _is_tool_enabled("traces"):
        tools.extend(_collect_tools(traces_cli.commands))

    # Scorers CLI tools (genai)
    if _is_tool_enabled("scorers"):
        tools.extend(_collect_tools(scorers_cli.commands))

    # Experiment tracking tools (genai)
    if _is_tool_enabled("experiments"):
        tools.extend(_collect_tools(mlflow.experiments.commands.commands))

    # Run management tools (genai)
    if _is_tool_enabled("runs"):
        tools.extend(_collect_tools(mlflow.runs.commands.commands))

    # Model serving tools (ml)
    if _is_tool_enabled("models"):
        tools.extend(_collect_tools(models_cli.commands.commands))

    # Deployment tools (ml)
    if _is_tool_enabled("deployments"):
        tools.extend(_collect_tools(deployments_cli.commands.commands))

    mcp = FastMCP(
        name="Mlflow MCP",
        tools=tools,
    )

    register_prompts(mcp)
    return mcp


def run_server() -> None:
    mcp = create_mcp()
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run_server()
