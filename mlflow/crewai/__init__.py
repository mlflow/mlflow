"""
The ``mlflow.crewai`` module provides an API for tracing CrewAI AI agents.
"""

import importlib
import logging

from packaging.version import Version

from mlflow.crewai.autolog import (
    patched_class_call,
    patched_standalone_call,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "crewai"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from CrewAI to MLflow.
    Note that asynchronous APIs and Tool calling are not recorded now.

    Args:
        log_traces: If ``True``, traces are logged for CrewAI agents.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the CrewAI autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during CrewAI
            autologging. If ``False``, show all events and warnings.
    """
    # TODO: Handle asynchronous tasks and crew executions
    import crewai

    CREWAI_VERSION = Version(crewai.__version__)

    class_method_map = {
        "crewai.Crew": ["kickoff", "kickoff_for_each", "train"],
        "crewai.Agent": ["execute_task"],
        "crewai.Task": ["execute_sync"],
        "crewai.LLM": ["call"],
        "crewai.Flow": ["kickoff"],
        "crewai.agents.agent_builder.base_agent_executor_mixin.CrewAgentExecutorMixin": [
            "_create_long_term_memory"
        ],
    }
    standalone_method_map = {}

    if CREWAI_VERSION >= Version("0.83.0"):
        # knowledge and memory are not available before 0.83.0
        class_method_map.update(
            {
                "crewai.memory.ShortTermMemory": ["save", "search"],
                "crewai.memory.LongTermMemory": ["save", "search"],
                "crewai.memory.EntityMemory": ["save", "search"],
                "crewai.Knowledge": ["query"],
            }
        )
        if CREWAI_VERSION < Version("0.157.0"):
            class_method_map.update({"crewai.memory.UserMemory": ["save", "search"]})

    # Modern Tool calling support for CrewAI >= 0.114.0
    if CREWAI_VERSION >= Version("0.114.0"):
        standalone_method_map.update(
            {"crewai.agents.crew_agent_executor": ["execute_tool_and_check_finality"]}
        )
    try:
        _apply_patches(standalone_method_map, _import_module, patched_standalone_call)
        _apply_patches(class_method_map, _import_class, patched_class_call)
    except (AttributeError, ModuleNotFoundError) as e:
        _logger.error("An exception happens when applying auto-tracing to crewai. Exception: %s", e)

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )


def _apply_patches(target_map, resolver, patch_fn):
    for target_path, methods in target_map.items():
        target = resolver(target_path)
        for method in methods:
            safe_patch(
                FLAVOR_NAME,
                target,
                method,
                patch_fn,
            )


def _import_module(module_path: str):
    return importlib.import_module(module_path)


def _import_class(class_path: str):
    *module_parts, class_name = class_path.rsplit(".", 1)
    module_path = ".".join(module_parts)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
