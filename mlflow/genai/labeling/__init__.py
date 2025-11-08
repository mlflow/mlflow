"""
Databricks Agent Labeling Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

from typing import Any

from mlflow.genai.labeling.databricks_utils import get_databricks_review_app
from mlflow.genai.labeling.labeling import Agent, LabelingSession, ReviewApp
from mlflow.genai.labeling.stores import _get_labeling_store


def get_review_app(experiment_id: str | None = None) -> "ReviewApp":
    """Gets or creates (if it doesn't exist) the review app for the given experiment ID.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Args:
        experiment_id: Optional. The experiment ID for which to get the review app. If not provided,
            the experiment ID is inferred from the current active environment.

    Returns:
        ReviewApp: The review app.
    """
    return ReviewApp(get_databricks_review_app(experiment_id))


def create_labeling_session(
    name: str,
    *,
    assigned_users: list[str] | None = None,
    agent: str | None = None,
    label_schemas: list[str] | None = None,
    enable_multi_turn_chat: bool = False,
    custom_inputs: dict[str, Any] | None = None,
) -> LabelingSession:
    """Create a new labeling session in the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

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
    store = _get_labeling_store()
    return store.create_labeling_session(
        name=name,
        assigned_users=assigned_users,
        agent=agent,
        label_schemas=label_schemas,
        enable_multi_turn_chat=enable_multi_turn_chat,
        custom_inputs=custom_inputs,
    )


def get_labeling_sessions() -> list[LabelingSession]:
    """Get all labeling sessions from the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Returns:
        list[LabelingSession]: The list of labeling sessions.
    """
    store = _get_labeling_store()
    return store.get_labeling_sessions()


def get_labeling_session(run_id: str) -> LabelingSession:
    """Get a labeling session from the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Args:
        run_id: The mlflow run ID of the labeling session to get.

    Returns:
        LabelingSession: The labeling session.
    """
    store = _get_labeling_store()
    return store.get_labeling_session(run_id)


def delete_labeling_session(labeling_session: LabelingSession):
    """Delete a labeling session from the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Args:
        labeling_session: The labeling session to delete.
    """
    store = _get_labeling_store()
    store.delete_labeling_session(labeling_session)

    # For backwards compatibility, return a ReviewApp instance only if using Databricks store
    from mlflow.genai.labeling.stores import DatabricksLabelingStore

    if isinstance(store, DatabricksLabelingStore):
        return ReviewApp(get_databricks_review_app())
    else:
        # For non-Databricks stores, we can't return a meaningful ReviewApp
        return None


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
