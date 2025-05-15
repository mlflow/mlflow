"""
Databricks Agent Review App Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>
"""

try:
    from databricks.agents.review_app import (
        Agent,
        get_review_app,
        label_schemas,
        LabelingSession,
        ReviewApp,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use mlflow.genai.review_app. "
        "Please install it with `pip install databricks-agents`."
    )

__all__ = [
    "Agent",
    "get_review_app",
    "label_schemas",
    "LabelingSession",
    "ReviewApp",
]
