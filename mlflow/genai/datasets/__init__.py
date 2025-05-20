"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

try:
    from databricks.agents.datasets import (
        Dataset as EvaluationDataset,
    )
except ImportError:
    logger.warning(
        "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
        "Please install it with `pip install databricks-agents`."
    )


def create_dataset(
    uc_table_name: str, experiment_id: Optional[Union[str, list[str]]] = None
) -> "EvaluationDataset":
    try:
        from databricks.agents.datasets import create_dataset
    except ImportError:
        raise ImportError(
            "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
            "Please install it with `pip install databricks-agents`."
        ) from None
    return create_dataset(uc_table_name, experiment_id)


def delete_dataset(uc_table_name: str) -> None:
    try:
        from databricks.agents.datasets import delete_dataset
    except ImportError:
        raise ImportError(
            "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
            "Please install it with `pip install databricks-agents`."
        ) from None
    return delete_dataset(uc_table_name)


def get_dataset(uc_table_name: str) -> "EvaluationDataset":
    try:
        from databricks.agents.datasets import get_dataset
    except ImportError:
        raise ImportError(
            "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
            "Please install it with `pip install databricks-agents`."
        ) from None
    return get_dataset(uc_table_name)


__all__ = [
    "EvaluationDataset",
    "create_dataset",
    "delete_dataset",
    "get_dataset",
]
