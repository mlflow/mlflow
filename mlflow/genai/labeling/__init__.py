"""
Databricks Agent Labeling Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

from typing import Any

from mlflow.genai.labeling.labeling import Agent, LabelingSession, ReviewApp

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.labeling`. "
    "Please install it with `pip install databricks-agents`."
)


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
    try:
        from databricks.agents.review_app import get_review_app as _get_review_app
    except ImportError:
        raise ImportError(_ERROR_MSG) from None

    return ReviewApp(_get_review_app(experiment_id))


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
    try:
        from databricks.agents.review_app import get_review_app as _get_review_app
    except ImportError:
        raise ImportError(_ERROR_MSG) from None

    return LabelingSession(
        _get_review_app().create_labeling_session(
            name=name,
            assigned_users=assigned_users or [],
            agent=agent,
            label_schemas=label_schemas or [],
            enable_multi_turn_chat=enable_multi_turn_chat,
            custom_inputs=custom_inputs,
        )
    )


def get_labeling_sessions() -> list[LabelingSession]:
    """Get all labeling sessions from the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Returns:
        list[LabelingSession]: The list of labeling sessions.
    """
    try:
        from databricks.agents.review_app import get_review_app as _get_review_app
    except ImportError:
        raise ImportError(_ERROR_MSG) from None

    return [LabelingSession(session) for session in _get_review_app().get_labeling_sessions()]


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
    labeling_sessions = get_labeling_sessions()
    labeling_session = next(
        (
            labeling_session
            for labeling_session in labeling_sessions
            if labeling_session.mlflow_run_id == run_id
        ),
        None,
    )
    if labeling_session is None:
        raise ValueError(f"Labeling session with run_id `{run_id}` not found")
    return LabelingSession(labeling_session)


def delete_labeling_session(labeling_session: LabelingSession) -> "ReviewApp":
    """Delete a labeling session from the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Args:
        labeling_session: The labeling session to delete.

    Returns:
        ReviewApp: The review app.
    """
    try:
        from databricks.agents.review_app import get_review_app as _get_review_app
    except ImportError:
        raise ImportError(_ERROR_MSG) from None

    return ReviewApp(_get_review_app().delete_labeling_session(labeling_session._session))


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
