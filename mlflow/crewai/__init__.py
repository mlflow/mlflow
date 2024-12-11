"""
The ``mlflow.crewai`` module provides an API for tracing CrewAI AI agents.
"""

import importlib
import logging

from packaging.version import Version

from mlflow.crewai.autolog import (
    patched_class_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "crewai"


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from CrewAI to MLflow.
    Note that asynchnorous APIs and Tool calling are not recorded now.

    Args:
        log_traces: If ``True``, traces are logged for CrewAI agents.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the CrewAI autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during CrewAI
            autologging. If ``False``, show all events and warnings.
    """
    # TODO: Handle asynchronous tasks and crew executions
    # TODO: Tool calling is not supported since the interface of tool in CrewAI is
    # changing drastically. Add patching once it's stabilized
    import crewai

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
    if Version(crewai.__version__) >= Version("0.83.0"):
        # knowledge and memory are not available before 0.83.0
        class_method_map.update(
            {
                "crewai.memory.ShortTermMemory": ["save", "search"],
                "crewai.memory.LongTermMemory": ["save", "search"],
                "crewai.memory.UserMemory": ["save", "search"],
                "crewai.memory.EntityMemory": ["save", "search"],
                "crewai.Knowledge": ["query"],
            }
        )
    try:
        for class_path, methods in class_method_map.items():
            *module_parts, class_name = class_path.rsplit(".", 1)
            module_path = ".".join(module_parts)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            for method in methods:
                safe_patch(
                    FLAVOR_NAME,
                    cls,
                    method,
                    patched_class_call,
                )
    except (AttributeError, ModuleNotFoundError) as e:
        _logger.error("An exception happens when applying auto-tracing to crewai. Exception: %s", e)
