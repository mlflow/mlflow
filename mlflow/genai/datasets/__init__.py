"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

import logging
from typing import Any

from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS
from mlflow.tracking import get_tracking_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import is_databricks_default_tracking_uri

_logger = logging.getLogger(__name__)

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
    "Please install it with `pip install databricks-agents`."
)


def _resolve_dataset_name(
    name: str | None = None,
    uc_table_name: str | None = None,
) -> str | None:
    """
    Helper to resolve dataset name from either 'name' or deprecated 'uc_table_name' parameter.

    Args:
        name: The dataset name parameter
        uc_table_name: The deprecated UC table name parameter

    Returns:
        The resolved dataset name

    Raises:
        ValueError: If both parameters are specified
    """
    if uc_table_name is not None:
        _logger.warning(
            "Parameter 'uc_table_name' is deprecated and will be removed in a future version. "
            "Please use 'name' parameter instead."
        )
        if name is None:
            return uc_table_name
        else:
            raise ValueError("Cannot specify both 'name' and 'uc_table_name' parameters.")
    return name


def _validate_databricks_params(
    name: str | None,
    dataset_id: str | None = None,
) -> None:
    """
    Validate parameters for Databricks environment.

    Args:
        name: The dataset name parameter (required)
        dataset_id: The dataset ID parameter (should not be provided)

    Raises:
        ValueError: If name is missing or dataset_id is provided
    """
    if name is None:
        raise ValueError("Parameter 'name' is required (or use deprecated 'uc_table_name').")
    if dataset_id is not None:
        raise ValueError(
            "Parameter 'dataset_id' is only supported outside of Databricks environments. "
            "Use 'name' parameter instead."
        )


def _validate_non_databricks_params(
    name: str | None,
    dataset_id: str | None = None,
) -> None:
    """
    Validate parameters for non-Databricks environment.

    Args:
        name: The dataset name parameter (should not be provided)
        dataset_id: The dataset ID parameter (required)

    Raises:
        ValueError: If dataset_id is missing or name is provided
    """
    if name is not None:
        raise ValueError(
            "Parameter 'name' is only supported in Databricks environments. "
            "Use 'dataset_id' parameter instead."
        )
    if dataset_id is None:
        raise ValueError(
            "Parameter 'dataset_id' is required. "
            "Use search_evaluation_datasets() to find the dataset ID by name if needed."
        )


def create_dataset(
    name: str | None = None,
    experiment_id: str | list[str] | None = None,
    tags: dict[str, Any] | None = None,
    *,
    uc_table_name: str | None = None,
) -> "EvaluationDataset":
    """
    Create a dataset with the given name and associate it with the given experiment.

    Args:
        name: The name of the dataset. In Databricks, this is the UC table name.
        experiment_id: The ID of the experiment(s) to associate the dataset with. If not provided,
            the current experiment is inferred from the environment.
        tags: Dictionary of tags to apply to the dataset. Not supported in Databricks.
        uc_table_name: (Deprecated) Use 'name' parameter instead. The UC table name of the dataset.

    Returns:
        An EvaluationDataset object representing the created dataset.
    """
    name = _resolve_dataset_name(name, uc_table_name)

    if name is None:
        raise ValueError("Parameter 'name' is required.")

    experiment_ids = [experiment_id] if isinstance(experiment_id, str) else experiment_id

    if is_databricks_default_tracking_uri(get_tracking_uri()):
        if tags is not None:
            raise NotImplementedError(
                "Tags are not supported in Databricks environments. "
                "Tags are managed through Unity Catalog."
            )
        try:
            from databricks.agents.datasets import create_dataset as db_create

            return EvaluationDataset(db_create(name, experiment_ids))
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e
    else:
        from mlflow.tracking.client import MlflowClient

        client = MlflowClient()
        return client.create_dataset(
            name=name,
            experiment_id=experiment_ids,
            tags=tags,
        )


def delete_dataset(
    name: str | None = None,
    dataset_id: str | None = None,
    *,
    uc_table_name: str | None = None,
) -> None:
    """
    Delete a dataset.

    Args:
        name: The name of the dataset (Databricks only). In Databricks, this is the UC table name.
        dataset_id: The ID of the dataset.
        uc_table_name: (Deprecated) Use 'name' parameter instead. The UC table name of the dataset.

    Note:
        - In Databricks environments: Use 'name' (or deprecated 'uc_table_name') to specify
            the dataset.
        - Outside of Databricks: Use 'dataset_id' to specify the dataset
    """
    # Handle deprecated parameter
    name = _resolve_dataset_name(name, uc_table_name)

    if is_databricks_default_tracking_uri(get_tracking_uri()):
        _validate_databricks_params(name, dataset_id)
        try:
            from databricks.agents.datasets import delete_dataset as db_delete

            return db_delete(name)
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e
    else:
        _validate_non_databricks_params(name, dataset_id)

        from mlflow.tracking.client import MlflowClient

        client = MlflowClient()
        client.delete_dataset(dataset_id)


def get_dataset(
    name: str | None = None,
    dataset_id: str | None = None,
    *,
    uc_table_name: str | None = None,
) -> "EvaluationDataset":
    """
    Get the dataset with the given name or ID.

    Args:
        name: The name of the dataset (Databricks only). In Databricks, this is the UC table name.
        dataset_id: The ID of the dataset.
        uc_table_name: (Deprecated) Use 'name' parameter instead. The UC table name of the dataset.

    Returns:
        An EvaluationDataset object representing the retrieved dataset.

    Note:
        - In Databricks environments: Use 'name' (or deprecated 'uc_table_name') to specify
            the dataset.
        - Outside of Databricks: Use 'dataset_id' to specify the dataset
    """
    name = _resolve_dataset_name(name, uc_table_name)

    if is_databricks_default_tracking_uri(get_tracking_uri()):
        _validate_databricks_params(name, dataset_id)
        try:
            from databricks.agents.datasets import get_dataset as db_get

            return EvaluationDataset(db_get(name))
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e
    else:
        _validate_non_databricks_params(name, dataset_id)

        from mlflow.tracking.client import MlflowClient

        client = MlflowClient()
        return client.get_dataset(dataset_id)


@experimental(version="3.3.0")
def search_datasets(
    experiment_ids: str | list[str] | None = None,
    filter_string: str | None = None,
    max_results: int = SEARCH_EVALUATION_DATASETS_MAX_RESULTS,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[EvaluationDataset]:
    """
    Search for datasets (non-Databricks only).

    Args:
        experiment_ids: Single experiment ID (str) or list of experiment IDs
        filter_string: Filter string for dataset names
        max_results: Maximum number of results
        order_by: Ordering criteria
        page_token: Token for next page of results

    Returns:
        PagedList of EvaluationDataset objects

    Note:
        This API is not available in Databricks environments.
    """
    if is_databricks_default_tracking_uri(get_tracking_uri()):
        raise NotImplementedError(
            "Dataset search is not available in Databricks. "
            "Use Unity Catalog search capabilities instead."
        )

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    return client.search_datasets(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


@experimental(version="3.3.0")
def set_dataset_tags(
    dataset_id: str,
    tags: dict[str, Any],
) -> None:
    """
    Set tags for a dataset.

    This implements a batch tag operation - existing tags are merged with new tags.
    To remove a tag, use delete_evaluation_dataset_tag() instead.

    Args:
        dataset_id: The ID of the dataset.
        tags: Dictionary of tags to set.

    Usage::

        set_dataset_tags(
            dataset_id="dataset_abc123",
            tags={
                "environment": "production",
                "version": "2.0",
            },
        )

    Note:
        This API is not available in Databricks environments yet.
    """
    if is_databricks_default_tracking_uri(get_tracking_uri()):
        raise NotImplementedError(
            "Dataset tag operations are not available in Databricks yet. "
            "Tags are managed through Unity Catalog."
        )

    if tags is None:
        raise ValueError("'tags' must be provided")

    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    client.set_dataset_tags(dataset_id, tags)


@experimental(version="3.3.0")
def delete_dataset_tag(
    dataset_id: str,
    key: str,
) -> None:
    """
    Delete a tag from a dataset.

    Args:
        dataset_id: The ID of the dataset.
        key: The tag key to delete.

    Usage::

        delete_dataset_tag(dataset_id="dataset_abc123", key="deprecated")

    Note:
        This API is not available in Databricks environments yet.
    """
    if is_databricks_default_tracking_uri(get_tracking_uri()):
        raise NotImplementedError(
            "Dataset tag operations are not available in Databricks yet. "
            "Tags are managed through Unity Catalog."
        )

    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    client.delete_dataset_tag(dataset_id, key)


__all__ = [
    "EvaluationDataset",
    "create_dataset",
    "delete_dataset",
    "delete_dataset_tag",
    "get_dataset",
    "search_datasets",
    "set_dataset_tags",
]
