"""
The ``mlflow.crewai`` module provides an API for tracing CrewAI AI agents.
"""

from mlflow.crewai.autolog import (
    patched_class_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

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
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for CrewAI agents.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the CrewAI autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during CrewAI
            autologging. If ``False``, show all events and warnings.
    """
    # TODO: handle asynchronous tasks and crew executions
    # TODO: interface of tool in CrewAI is changing drastically. Add patching once it's stabilized
    import crewai

    safe_patch(
        FLAVOR_NAME,
        crewai.Crew,
        "kickoff",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        crewai.Crew,
        "kickoff_for_each",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        crewai.Crew,
        "train",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        crewai.Agent,
        "execute_task",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        crewai.Task,
        "execute_sync",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        crewai.LLM,
        "call",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        crewai.Flow,
        "kickoff",
        patched_class_call,
    )
    try:
        # knowledge and memory are not available before 0.83.0
        safe_patch(
            FLAVOR_NAME,
            crewai.memory.ShortTermMemory,
            "save",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.ShortTermMemory,
            "search",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.ShortTermMemory,
            "reset",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.LongTermMemory,
            "save",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.LongTermMemory,
            "search",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.LongTermMemory,
            "reset",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.UserMemory,
            "save",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.UserMemory,
            "search",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.EntityMemory,
            "search",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.EntityMemory,
            "save",
            patched_class_call,
        )

        safe_patch(
            FLAVOR_NAME,
            crewai.memory.EntityMemory,
            "reset",
            patched_class_call,
        )
        safe_patch(
            FLAVOR_NAME,
            crewai.Knowledge,
            "query",
            patched_class_call,
        )
    except AttributeError:
        pass
