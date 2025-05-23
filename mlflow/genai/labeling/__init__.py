"""
Databricks Agent Labeling Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

from typing import Any, Optional

try:
    from databricks.agents.review_app import (
        Agent,
        LabelingSession,
        ReviewApp,
        get_review_app,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use `mlflow.genai.labeling`. "
        "Please install it with `pip install databricks-agents`."
    ) from None


def create_labeling_session(
    name: str,
    *,
    assigned_users: list[str] = [],  # noqa: B006
    agent: Optional[str] = None,
    label_schemas: list[str] = [],  # noqa: B006
    enable_multi_turn_chat: bool = False,
    custom_inputs: Optional[dict[str, Any]] = None,
) -> LabelingSession:
    """Create a new labeling session in the review app.

    Args:
        name: The name of the labeling session.
        assigned_users: The users that will be assigned to label items in the session.
        agent: The agent to be used to generate responses for the items in the session.
        label_schemas: The label schemas to be used in the session.
        enable_multi_turn_chat: Whether to enable multi-turn chat labeling for the session.
        custom_inputs: Optional. Custom inputs to be used in the session.

    Returns:
        LabelingSession: The created labeling session.
    """
    return get_review_app().create_labeling_session(
        name=name,
        assigned_users=assigned_users,
        agent=agent,
        label_schemas=label_schemas,
        enable_multi_turn_chat=enable_multi_turn_chat,
        custom_inputs=custom_inputs,
    )


def get_labeling_sessions() -> list[LabelingSession]:
    """Get all labeling sessions from the review app.

    Returns:
        list[LabelingSession]: The list of labeling sessions.
    """
    return get_review_app().get_labeling_sessions()


def get_labeling_session(name: str) -> LabelingSession:
    """Get a labeling session from the review app.

    Args:
        name: The name of the labeling session to get.

    Returns:
        LabelingSession: The labeling session.
    """
    labeling_sessions = get_labeling_sessions()
    labeling_session = next(
        (
            labeling_session
            for labeling_session in labeling_sessions
            if labeling_session.name == name
        ),
        None,
    )
    if labeling_session is None:
        raise ValueError(f"Labeling session with name `{name}` not found")
    return labeling_session


def delete_labeling_session(labeling_session: LabelingSession) -> ReviewApp:
    """Delete a labeling session from the review app.

    Args:
        labeling_session: The labeling session to delete.

    Returns:
        ReviewApp: The review app.
    """
    return get_review_app().delete_labeling_session(labeling_session)


__all__ = [
    "Agent",
    "LabelingSession",
    "ReviewApp",
    "get_review_app",
    "create_labeling_session",
    "get_labeling_sessions",
    "get_labeling_session",
    "delete_labeling_session",
]
