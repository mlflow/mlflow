"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>
"""

try:
    from databricks.agents.datasets import (
        create_dataset,
        Dataset,
        delete_dataset,
        get_dataset,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use mlflow.genai.datasets. "
        "Please install it with `pip install databricks-agents`."
    )

__all__ = [
    "create_dataset",
    "Dataset",
    "delete_dataset",
    "get_dataset",
]
