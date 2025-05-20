"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

try:
    from databricks.agents.datasets import (
        Dataset as EvaluationDataset,
    )
    from databricks.agents.datasets import (
        create_dataset,
        delete_dataset,
        get_dataset,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
        "Please install it with `pip install databricks-agents`."
    )

__all__ = [
    "EvaluationDataset",
    "create_dataset",
    "delete_dataset",
    "get_dataset",
]
