"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS
from mlflow.tracking import get_tracking_uri
from mlflow.utils.annotations import deprecated_parameter, experimental
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri

_logger = logging.getLogger(__name__)

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
    "Please install it with `pip install databricks-agents`."
)

_DATABRICKS_CONFIG_PROFILE_ENV_VAR = "DATABRICKS_CONFIG_PROFILE"


@contextmanager
def _databricks_profile_env():
    """
    Context manager that temporarily sets DATABRICKS_CONFIG_PROFILE based on the tracking URI.

    This ensures that databricks.agents SDK functions use the correct profile specified
    in the MLflow tracking URI. The databricks.agents SDK creates WorkspaceClient instances
    internally without accepting profile parameters, so it relies on the
    DATABRICKS_CONFIG_PROFILE environment variable to determine which profile to use.

    The tracking URI profile takes precedence over any existing DATABRICKS_CONFIG_PROFILE
    environment variable for the duration of MLflow operations. The original value is
    restored after the operation completes.
    """
    tracking_uri = get_tracking_uri()
    profile, _ = get_db_info_from_uri(tracking_uri)

    if not profile:
        yield
        return

    original_profile = os.environ.get(_DATABRICKS_CONFIG_PROFILE_ENV_VAR)
    os.environ[_DATABRICKS_CONFIG_PROFILE_ENV_VAR] = profile

    try:
        yield
    finally:
        if original_profile is not None:
            os.environ[_DATABRICKS_CONFIG_PROFILE_ENV_VAR] = original_profile
        else:
            os.environ.pop(_DATABRICKS_CONFIG_PROFILE_ENV_VAR, None)


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
            "Use search_datasets() to find the dataset ID by name if needed."
        )


@deprecated_parameter("uc_table_name", "name")
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

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import create_dataset

            # Create a dataset with a single experiment
            dataset = create_dataset(
                name="customer_support_qa_v1",
                experiment_id="0",  # Default experiment
                tags={
                    "version": "1.0",
                    "purpose": "regression_testing",
                    "model": "gpt-4",
                    "team": "ml-platform",
                },
            )
            print(f"Created dataset: {dataset.dataset_id}")
            # Output: Created dataset: d-1a2b3c4d5e6f7890abcdef1234567890

            # Create a dataset linked to multiple experiments
            multi_exp_dataset = create_dataset(
                name="cross_team_eval_dataset",
                experiment_id=["1", "2", "5"],  # Multiple experiment IDs
                tags={
                    "coverage": "comprehensive",
                    "status": "development",
                },
            )

            # Create a dataset without tags (minimal example)
            simple_dataset = create_dataset(
                name="quick_test_dataset",
                experiment_id="3",  # Specific experiment
            )
    """
    if name is None:
        raise ValueError("Parameter 'name' is required.")

    experiment_ids = [experiment_id] if isinstance(experiment_id, str) else experiment_id

    if is_databricks_uri(get_tracking_uri()):
        if tags is not None:
            raise NotImplementedError(
                "Tags are not supported in Databricks environments. "
                "Tags are managed through Unity Catalog."
            )
        try:
            from databricks.agents.datasets import create_dataset as db_create

            with _databricks_profile_env():
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

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import delete_dataset, search_datasets

            # Delete a specific dataset by ID (non-Databricks)
            delete_dataset(dataset_id="d-4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e")

            # Clean up old test datasets
            test_datasets = search_datasets(
                filter_string="name LIKE 'test_%' AND tags.environment = 'development'",
                order_by=["created_time ASC"],
            )

            # Delete datasets older than the most recent 5
            if len(test_datasets) > 5:
                for dataset in test_datasets[:-5]:  # Keep the 5 most recent
                    print(f"Deleting old test dataset: {dataset.name}")
                    delete_dataset(dataset_id=dataset.dataset_id)

            # Delete datasets with specific criteria
            deprecated_datasets = search_datasets(filter_string="tags.status = 'deprecated'")
            for dataset in deprecated_datasets:
                delete_dataset(dataset_id=dataset.dataset_id)
                print(f"Deleted deprecated dataset: {dataset.name}")

    .. warning::
        Deleting a dataset is permanent and cannot be undone. All associated
        records, tags, and metadata will be permanently removed.
    """

    if is_databricks_uri(get_tracking_uri()):
        _validate_databricks_params(name, dataset_id)
        try:
            from databricks.agents.datasets import delete_dataset as db_delete

            with _databricks_profile_env():
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

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import get_dataset

            # Get a dataset by ID (non-Databricks)
            dataset = get_dataset(dataset_id="d-7f2e3a9b8c1d4e5f6a7b8c9d0e1f2a3b")

            # Access dataset properties
            print(f"Dataset name: {dataset.name}")
            print(f"Tags: {dataset.tags}")
            print(f"Created by: {dataset.created_by}")

            # Work with the dataset
            df = dataset.to_df()  # Convert to pandas DataFrame
            schema = dataset.schema  # Get auto-computed schema
            profile = dataset.profile  # Get dataset statistics

            # Add new records to the dataset
            new_test_cases = [
                {
                    "inputs": {"question": "What is MLflow?"},
                    "expectations": {"accuracy": 0.95, "contains_tracking": True},
                }
            ]
            dataset.merge_records(new_test_cases)
    """

    if is_databricks_uri(get_tracking_uri()):
        _validate_databricks_params(name, dataset_id)
        try:
            from databricks.agents.datasets import get_dataset as db_get

            with _databricks_profile_env():
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

    .. warning::
        Calling ``search_datasets()`` without any parameters will return ALL datasets
        in your tracking server. This can be slow or even crash your Python session if
        you have many datasets. Always use filters or ``max_results`` to limit the results.

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

    .. list-table:: Common Search Patterns
        :widths: 40 60
        :header-rows: 1

        * - Search Pattern
          - Example Code

        * - **Find datasets by name**
          - .. code-block:: python

                # Exact match
                datasets = search_datasets(
                    filter_string="name = 'production_qa_v2'"
                )

                # Pattern matching
                datasets = search_datasets(
                    filter_string="name LIKE 'qa_%'"
                )

        * - **Find datasets by experiment**
          - .. code-block:: python

                # Single experiment
                datasets = search_datasets(
                    experiment_ids="1"
                )

                # Multiple experiments
                datasets = search_datasets(
                    experiment_ids=["0", "1", "2", "5"]
                )

        * - **Find datasets by tags**
          - .. code-block:: python

                # Single tag
                datasets = search_datasets(
                    filter_string="tags.environment = 'production'"
                )

                # Multiple tags with AND
                datasets = search_datasets(
                    filter_string="tags.status = 'validated' AND tags.version = '2.0'"
                )

        * - **Find datasets by creator**
          - .. code-block:: python

                datasets = search_datasets(
                    filter_string="created_by = 'alice@company.com'"
                )

        * - **Find recent datasets**
          - .. code-block:: python

                # Last 10 datasets created
                datasets = search_datasets(
                    order_by=["created_time DESC"],
                    max_results=10
                )

        * - **Complex search**
          - .. code-block:: python

                # Production-ready datasets from specific team
                datasets = search_datasets(
                    experiment_ids="1",
                    filter_string="tags.status = 'production' AND "
                                  "tags.team = 'ml-platform' AND "
                                  "name LIKE '%customer%'",
                    order_by=["last_update_time DESC"],
                    max_results=20
                )

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import search_datasets

            # WARNING: This returns ALL datasets - use with caution!
            # all_datasets = search_datasets()  # May be slow or crash

            # Better: Always use filters or limits
            recent_datasets = search_datasets(max_results=100)

            # Search in specific experiments
            exp_datasets = search_datasets(experiment_ids=["1", "2", "3"])

            # Find production datasets
            prod_datasets = search_datasets(
                filter_string="tags.environment = 'production'", order_by=["name ASC"]
            )

            # Iterate through results (pagination handled automatically)
            for dataset in prod_datasets:
                print(f"{dataset.name} (ID: {dataset.dataset_id})")
                print(f"  Tags: {dataset.tags}")

    Note:
        This API is not available in Databricks environments. Use Unity Catalog
        search capabilities in Databricks instead.
    """
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    # Check if we're using Databricks - don't set defaults for unsupported parameters
    is_databricks = is_databricks_uri(get_tracking_uri())

    # Set default filter to return datasets created in the last 7 days if no filter provided
    # Skip this for Databricks as filter_string is not supported
    # Also handle empty list/string cases where user might pass [] or ""
    if not is_databricks and not filter_string:
        # 7 days ago in milliseconds
        seven_days_ago = int((time.time() - 7 * 24 * 60 * 60) * 1000)
        filter_string = f"created_time >= {seven_days_ago}"

    # Set default order by creation time DESC if no order provided
    # Skip this for Databricks as order_by is not supported
    if not is_databricks and order_by is None:
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
    To remove a tag, set its value to None or use delete_dataset_tag() instead.

    Args:
        dataset_id: The ID of the dataset.
        tags: Dictionary of tags to set. Setting a value to None removes the tag.

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import set_dataset_tags, get_dataset

            # Get your dataset
            dataset = get_dataset(dataset_id="d-8f3a2b1c4e5d6f7a8b9c0d1e2f3a4b5c")

            # Add or update multiple tags
            set_dataset_tags(
                dataset_id=dataset.dataset_id,
                tags={
                    "environment": "production",  # Add new tag
                    "version": "2.0",  # Update existing tag
                    "validated": "true",
                    "validation_date": "2024-11-01",
                    "team": "ml-platform",
                },
            )

            # Remove tags by setting to None
            set_dataset_tags(
                dataset_id=dataset.dataset_id,
                tags={
                    "deprecated_tag": None,  # This removes the tag
                    "old_version": None,  # This also removes the tag
                },
            )

            # Update status after validation
            set_dataset_tags(
                dataset_id=dataset.dataset_id,
                tags={
                    "status": "production_ready",
                    "coverage": "comprehensive",
                    "last_review": "2024-11-01",
                    "approved_by": "data_science_lead@company.com",
                },
            )

    Note:
        This API is not available in Databricks environments yet.
        Tags in Databricks are managed through Unity Catalog.
    """
    if is_databricks_uri(get_tracking_uri()):
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

    Examples:
        .. code-block:: python

            from mlflow.genai.datasets import delete_dataset_tag, get_dataset

            # Get your dataset
            dataset = get_dataset(dataset_id="d-9e8f7c6b5a4d3e2f1a0b9c8d7e6f5a4b")

            # Remove a single tag
            delete_dataset_tag(dataset_id=dataset.dataset_id, key="deprecated")

            # Remove outdated tags during cleanup
            outdated_tags = ["old_version", "temp_flag", "development_only"]
            for tag_key in outdated_tags:
                delete_dataset_tag(dataset_id=dataset.dataset_id, key=tag_key)

            # Check remaining tags
            updated_dataset = get_dataset(dataset_id=dataset.dataset_id)
            print(f"Remaining tags: {updated_dataset.tags}")

    Note:
        This API is not available in Databricks environments yet.
        Tags in Databricks are managed through Unity Catalog.
    """
    if is_databricks_uri(get_tracking_uri()):
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

    if is_databricks_uri(get_tracking_uri()):
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
