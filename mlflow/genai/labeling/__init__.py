"""
Databricks Agent Labeling Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

try:
    from databricks.agents.review_app import (
        Agent,
        LabelingSession,
        ReviewApp,
        get_review_app,
        label_schemas,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use mlflow.genai.labeling. "
        "Please install it with `pip install databricks-agents`."
    )

__all__ = [
    "Agent",
    "get_review_app",
    "label_schemas",
    "LabelingSession",
    "ReviewApp",
]
