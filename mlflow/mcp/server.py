import contextlib
import io
import os
from typing import TYPE_CHECKING, Any, Callable

import click
from click.types import BOOL, FLOAT, INT, STRING, UUID

import mlflow.experiments
import mlflow.runs
import mlflow.store.artifact.cli
from mlflow.ai_commands.ai_command_utils import get_command_body, list_commands
from mlflow.cli.scorers import commands as scorers_cli
from mlflow.cli.traces import commands as traces_cli

# Environment variable to control which tool categories are enabled
# Comma-separated list: "traces,scorers,experiments,runs,artifacts,models,deployments"
# Use "all" to enable all tools (default)
MLFLOW_MCP_TOOLS = os.environ.get("MLFLOW_MCP_TOOLS", "all")

# Optional CLI groups
try:
    import mlflow.models.cli as models_cli

    _MODELS_CLI_AVAILABLE = True
except ImportError:
    _MODELS_CLI_AVAILABLE = False

try:
    import mlflow.deployments.cli as deployments_cli

    _DEPLOYMENTS_CLI_AVAILABLE = True
except ImportError:
    _DEPLOYMENTS_CLI_AVAILABLE = False

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
        # Capture stdout and stderr
        string_io = io.StringIO()
        with (
            contextlib.redirect_stdout(string_io),
            contextlib.redirect_stderr(string_io),
        ):
            # Fill in defaults for missing optional arguments
            for param in command.params:
                if param.name not in kwargs:
                    kwargs[param.name] = param.default
            command.callback(**kwargs)  # type: ignore[misc]
        return string_io.getvalue().strip()

    return wrapper


def cmd_to_function_tool(cmd: click.Command, suffix: str = "") -> "FunctionTool":
    """
    Converts a Click command to a FunctionTool.

    Args:
        cmd: The Click command to convert.
        suffix: Optional suffix to add to the tool name (e.g., "experiments").
    """
    from fastmcp.tools import FunctionTool

    # Use the command name (CLI name) with optional suffix
    base_name = cmd.name or (cmd.callback.__name__ if cmd.callback else "unknown")
    name = f"{base_name}_{suffix}" if suffix else base_name
    return FunctionTool(
        fn=fn_wrapper(cmd),
        name=name,
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
    if MLFLOW_MCP_TOOLS.lower() == "all":
        return True
    enabled_tools = [t.strip().lower() for t in MLFLOW_MCP_TOOLS.split(",")]
    return category.lower() in enabled_tools


def create_mcp() -> "FastMCP":
    from fastmcp import FastMCP

    tools: list = []

    # Traces CLI tools
    if _is_tool_enabled("traces"):
        tools.extend(cmd_to_function_tool(cmd, "traces") for cmd in traces_cli.commands.values())

    # Scorers CLI tools
    if _is_tool_enabled("scorers"):
        tools.extend(cmd_to_function_tool(cmd, "scorers") for cmd in scorers_cli.commands.values())

    # Experiment tracking tools
    if _is_tool_enabled("experiments"):
        tools.extend(
            cmd_to_function_tool(cmd, "experiments")
            for cmd in mlflow.experiments.commands.commands.values()
        )

    # Run management tools
    if _is_tool_enabled("runs"):
        tools.extend(
            cmd_to_function_tool(cmd, "runs") for cmd in mlflow.runs.commands.commands.values()
        )

    # Artifact handling tools
    if _is_tool_enabled("artifacts"):
        tools.extend(
            cmd_to_function_tool(cmd, "artifacts")
            for cmd in mlflow.store.artifact.cli.commands.commands.values()
        )

    # Model serving tools
    if _is_tool_enabled("models") and _MODELS_CLI_AVAILABLE:
        tools.extend(
            cmd_to_function_tool(cmd, "models") for cmd in models_cli.commands.commands.values()
        )

    # Deployment tools
    if _is_tool_enabled("deployments") and _DEPLOYMENTS_CLI_AVAILABLE:
        tools.extend(
            cmd_to_function_tool(cmd, "deployments")
            for cmd in deployments_cli.commands.commands.values()
        )

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
