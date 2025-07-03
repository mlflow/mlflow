"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

from typing import Optional, Union

from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
    "Please install it with `pip install databricks-agents`."
)


def create_dataset(
    uc_table_name: str, experiment_id: Optional[Union[str, list[str]]] = None
) -> "EvaluationDataset":
    """Create a dataset with the given name and associate it with the given experiment.

    Args:
        uc_table_name: The UC table name of the dataset.
        experiment_id: The ID of the experiment to associate the dataset with. If not provided,
            the current experiment is inferred from the environment.
    """
    try:
        from databricks.agents.datasets import create_dataset
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return EvaluationDataset(create_dataset(uc_table_name, experiment_id))


def delete_dataset(uc_table_name: str) -> None:
    """Delete the dataset with the given name.

    Args:
        uc_table_name: The UC table name of the dataset.
    """
    try:
        from databricks.agents.datasets import delete_dataset
    except ImportError:
        raise ImportError(_ERROR_MSG) from None
    return delete_dataset(uc_table_name)


def get_dataset(uc_table_name: str) -> "EvaluationDataset":
    """Get the dataset with the given name.

    Args:
        uc_table_name: The UC table name of the dataset.
    """
    try:
        from databricks.agents.datasets import get_dataset
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return EvaluationDataset(get_dataset(uc_table_name))


__all__ = [
    "create_dataset",
    "delete_dataset",
    "get_dataset",
    "EvaluationDataset",
]
