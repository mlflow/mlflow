"""
Databricks utilities for MLflow GenAI labeling functionality.
"""

_ERROR_MSG = (
    "The `databricks-agents` package is required to use labeling functionality. "
    "Please install it with `pip install databricks-agents`."
)


def get_databricks_review_app(experiment_id: str | None = None):
    """Import databricks.agents.review_app and return a review app instance."""
    try:
        from databricks.agents import review_app
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e

    return review_app.get_review_app(experiment_id)
