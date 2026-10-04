"""
Typed MLflow MCP tools shared by the stdio server (``mlflow mcp run``) and the tracking
server's HTTP endpoint (``mlflow server --enable-mcp``).

Each tool validates its inputs, calls the tracking store resolved from MLflow's configured
tracking URI, and returns a pydantic model. The package does not depend on fastmcp: the
transports register these functions, and the tracking server wraps them with per-tool
authorization.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from mlflow.mcp.tools import experiments, runs, scorers, traces

ToolCategory = Literal["traces", "scorers", "experiments", "runs"]


@dataclass(frozen=True)
class SharedTool:
    """
    Args:
        fn: The tool implementation. Its name is the tool name, its docstring the description
            and its annotations the input and output schemas.
        category: The ``MLFLOW_MCP_TOOLS`` category the tool belongs to.
        read_only: Whether the tool leaves the store unchanged.
        destructive: Whether the tool deletes data.
    """

    fn: Callable[..., Any]
    category: ToolCategory
    read_only: bool = False
    destructive: bool = False

    @property
    def name(self) -> str:
        return self.fn.__name__


SHARED_TOOLS: tuple[SharedTool, ...] = (
    # Traces
    SharedTool(traces.search_traces, "traces", read_only=True),
    SharedTool(traces.get_trace, "traces", read_only=True),
    SharedTool(traces.delete_traces, "traces", destructive=True),
    SharedTool(traces.set_trace_tag, "traces"),
    SharedTool(traces.delete_trace_tag, "traces", destructive=True),
    SharedTool(traces.log_trace_feedback, "traces"),
    SharedTool(traces.log_trace_expectation, "traces"),
    SharedTool(traces.get_trace_assessment, "traces", read_only=True),
    SharedTool(traces.update_trace_assessment, "traces"),
    SharedTool(traces.delete_trace_assessment, "traces", destructive=True),
    # Scorers
    SharedTool(scorers.list_scorers, "scorers", read_only=True),
    SharedTool(scorers.register_llm_judge_scorer, "scorers"),
    # Experiments
    SharedTool(experiments.create_experiment, "experiments"),
    SharedTool(experiments.update_experiment, "experiments"),
    SharedTool(experiments.search_experiments, "experiments", read_only=True),
    SharedTool(experiments.get_experiment, "experiments", read_only=True),
    SharedTool(experiments.delete_experiment, "experiments", destructive=True),
    SharedTool(experiments.restore_experiment, "experiments"),
    SharedTool(experiments.rename_experiment, "experiments"),
    # Runs
    SharedTool(runs.list_runs, "runs", read_only=True),
    SharedTool(runs.delete_run, "runs", destructive=True),
    SharedTool(runs.restore_run, "runs"),
    SharedTool(runs.describe_run, "runs", read_only=True),
    SharedTool(runs.create_run, "runs"),
    SharedTool(runs.link_traces_to_run, "runs"),
)
