"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

import logging
import time
from typing import Any

from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS
from mlflow.telemetry.events import CreateDatasetEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracking import get_tracking_uri
from mlflow.utils.annotations import deprecated_parameter, experimental
from mlflow.utils.databricks_utils import is_databricks_default_tracking_uri

_logger = logging.getLogger(__name__)

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
    "Please install it with `pip install databricks-agents`."
)


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
        raise ValueError("Parameter 'name' is required.")
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


@deprecated_parameter("uc_table_name", "name")
@record_usage_event(CreateDatasetEvent)
def create_dataset(
    name: str | None = None,
    experiment_id: str | list[str] | None = None,
    tags: dict[str, Any] | None = None,
) -> "EvaluationDataset":
    """
    Create a dataset with the given name and associate it with the given experiment.

    Args:
        name: The name of the dataset. In Databricks, this is the UC table name.
        experiment_id: The ID of the experiment(s) to associate the dataset with. If not provided,
            the current experiment is inferred from the environment.
        tags: Dictionary of tags to apply to the dataset. Not supported in Databricks.

    Returns:
        An EvaluationDataset object representing the created dataset.
    """
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

        if experiment_ids is None:
            from mlflow.tracking.fluent import _get_experiment_id

            current_exp_id = _get_experiment_id()
            if current_exp_id:
                experiment_ids = [current_exp_id]

        mlflow_dataset = MlflowClient().create_dataset(
            name=name,
            experiment_id=experiment_ids,
            tags=tags,
        )
        return EvaluationDataset(mlflow_dataset)


@deprecated_parameter("uc_table_name", "name")
def delete_dataset(
    name: str | None = None,
    dataset_id: str | None = None,
) -> None:
    """
    Delete a dataset.

    Args:
        name: The name of the dataset (Databricks only). In Databricks, this is the UC table name.
        dataset_id: The ID of the dataset.

    Note:
        - In Databricks environments: Use 'name' to specify the dataset.
        - Outside of Databricks: Use 'dataset_id' to specify the dataset
    """

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

        MlflowClient().delete_dataset(dataset_id)


@deprecated_parameter("uc_table_name", "name")
def get_dataset(
    name: str | None = None,
    dataset_id: str | None = None,
) -> "EvaluationDataset":
    """
    Get the dataset with the given name or ID.

    Args:
        name: The name of the dataset (Databricks only). In Databricks, this is the UC table name.
        dataset_id: The ID of the dataset.

    Returns:
        An EvaluationDataset object representing the retrieved dataset.

    Note:
        - In Databricks environments: Use 'name' to specify the dataset.
        - Outside of Databricks: Use 'dataset_id' to specify the dataset
    """

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

        mlflow_dataset = MlflowClient().get_dataset(dataset_id)
        return EvaluationDataset(mlflow_dataset)


@experimental(version="3.4.0")
def search_datasets(
    experiment_ids: str | list[str] | None = None,
    filter_string: str | None = None,
    max_results: int | None = None,
    order_by: list[str] | None = None,
) -> list[EvaluationDataset]:
    """
    Search for datasets (non-Databricks only).

    Args:
        experiment_ids: Single experiment ID (str) or list of experiment IDs to filter by.
            If None, searches across all experiments.
        filter_string: SQL-like filter string for dataset attributes. If not specified,
            defaults to filtering for datasets created in the last 7 days. Supports filtering by:
            - name: Dataset name
            - created_by: User who created the dataset
            - last_updated_by: User who last updated the dataset
            - created_time: Creation timestamp (milliseconds since epoch)
            - tags.<key>: Tag values
        max_results: Maximum number of results. If not specified, returns all datasets.
        order_by: List of columns to order by. Each entry can include an optional
            "DESC" or "ASC" suffix (default is "ASC"). If not specified, defaults to
            ["created_time DESC"]. Supported columns:
            - name
            - created_time
            - last_update_time

    Returns:
        List of EvaluationDataset objects matching the search criteria

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import search_datasets

            # Search all datasets
            all_datasets = search_datasets()

            # Search datasets in specific experiments
            exp_datasets = search_datasets(experiment_ids=["exp1", "exp2"])

            # Search by name pattern
            qa_datasets = search_datasets(filter_string="name LIKE 'qa_%'")

            # Search by creator
            user_datasets = search_datasets(filter_string="created_by = 'alice@company.com'")

            # Search by tags
            prod_datasets = search_datasets(filter_string="tags.environment = 'production'")

            # Complex filter with AND condition
            recent_prod = search_datasets(
                filter_string="tags.environment = 'production' AND tags.version >= '2.0'"
            )

            # Order by creation time (newest first)
            recent_datasets = search_datasets(order_by=["created_time DESC"])

            # Combine multiple search criteria
            filtered_datasets = search_datasets(
                experiment_ids="exp123",
                filter_string="name LIKE 'eval_%' AND tags.status = 'validated'",
                order_by=["last_update_time DESC", "name ASC"],
                max_results=10,
            )

    Note:
        This API is not available in Databricks environments. Use Unity Catalog
        search capabilities in Databricks instead.
    """
    if is_databricks_default_tracking_uri(get_tracking_uri()):
        raise NotImplementedError(
            "Dataset search is not available in Databricks. "
            "Use Unity Catalog search capabilities instead."
        )

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    # Set default filter to return datasets created in the last 7 days if no filter provided
    # Also handle empty list/string cases where user might pass [] or ""
    if not filter_string:
        # 7 days ago in milliseconds
        seven_days_ago = int((time.time() - 7 * 24 * 60 * 60) * 1000)
        filter_string = f"created_time >= {seven_days_ago}"

    # Set default order by creation time DESC if no order provided
    if order_by is None:
        order_by = ["created_time DESC"]

    from mlflow.tracking.client import MlflowClient
    from mlflow.utils import get_results_from_paginated_fn

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_datasets(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=number_to_get,
            order_by=order_by,
            page_token=next_page_token,
        )

    mlflow_datasets = get_results_from_paginated_fn(
        pagination_wrapper_func,
        SEARCH_EVALUATION_DATASETS_MAX_RESULTS,
        max_results,
    )
    return [EvaluationDataset(dataset) for dataset in mlflow_datasets]


@experimental(version="3.4.0")
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

    MlflowClient().set_dataset_tags(dataset_id, tags)


@experimental(version="3.4.0")
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

    MlflowClient().delete_dataset_tag(dataset_id, key)


def _validate_association_operation():
    """Validate that dataset association operations can be performed."""
    from mlflow.store.tracking.file_store import FileStore
    from mlflow.tracking._tracking_service.utils import _get_store

    if is_databricks_default_tracking_uri(get_tracking_uri()):
        raise NotImplementedError(
            "Dataset association operations are not available in Databricks yet. "
            "Associations are managed through Unity Catalog."
        )

    store = _get_store()
    if isinstance(store, FileStore):
        raise NotImplementedError(
            "Dataset association operations are not supported with FileStore backend. "
            "Please use a database-backed tracking store."
        )


def add_dataset_to_experiments(dataset_id: str, experiment_ids: list[str]) -> "EvaluationDataset":
    """
    Add a dataset to additional experiments.

    This allows reusing datasets across multiple experiments for evaluation purposes.

    Args:
        dataset_id: The ID of the dataset to update.
        experiment_ids: List of experiment IDs to associate with the dataset.

    Returns:
        The updated EvaluationDataset with new experiment associations.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.datasets import add_dataset_to_experiments

            # Add dataset to new experiments
            dataset = add_dataset_to_experiments(
                dataset_id="d-abc123", experiment_ids=["1", "2", "3"]
            )
            print(f"Dataset now associated with {len(dataset.experiment_ids)} experiments")
    """
    _validate_association_operation()

    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    mlflow_dataset = client.add_dataset_to_experiments(dataset_id, experiment_ids)
    return EvaluationDataset(mlflow_dataset)


def remove_dataset_from_experiments(
    dataset_id: str, experiment_ids: list[str]
) -> "EvaluationDataset":
    """
    Remove a dataset from experiments.

    This operation is idempotent - removing non-existent associations will not raise an error
    but will issue a warning.

    Args:
        dataset_id: The ID of the dataset to update.
        experiment_ids: List of experiment IDs to disassociate from the dataset.

    Returns:
        The updated EvaluationDataset after removing experiment associations.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.datasets import remove_dataset_from_experiments

            # Remove dataset from experiments
            dataset = remove_dataset_from_experiments(
                dataset_id="d-abc123", experiment_ids=["1", "2"]
            )
            print(f"Dataset now associated with {len(dataset.experiment_ids)} experiments")
    """
    _validate_association_operation()

    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    mlflow_dataset = client.remove_dataset_from_experiments(dataset_id, experiment_ids)
    return EvaluationDataset(mlflow_dataset)


__all__ = [
    "EvaluationDataset",
    "add_dataset_to_experiments",
    "create_dataset",
    "delete_dataset",
    "delete_dataset_tag",
    "get_dataset",
    "remove_dataset_from_experiments",
    "search_datasets",
    "set_dataset_tags",
]
