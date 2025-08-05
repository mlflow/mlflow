"""
Databricks Agent Datasets Python SDK. For more details see Databricks Agent Evaluation:
 <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#datasets>
"""

import logging
from typing import Any, Optional, Union

from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import deprecated, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime

_logger = logging.getLogger(__name__)

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.datasets`. "
    "Please install it with `pip install databricks-agents`."
)


@experimental(version="3.3.0")
def create_evaluation_dataset(
    name: str,
    experiment_ids: Optional[Union[str, list[str]]] = None,
    tags: Optional[dict[str, Any]] = None,
) -> EvaluationDataset:
    """
    Create an empty evaluation dataset. Use merge_records() to add data.

    Args:
        name: Dataset name. In Databricks, this is the UC table name.
        experiment_ids: Single experiment ID (str) or list of experiment IDs.
        tags: Dictionary of tags to apply to the dataset. Not available in Databricks.
            To set the dataset creator, include {"mlflow.user": "username"} in tags.

    OSS Usage::

        dataset = create_evaluation_dataset(
            name="my_dataset",
            experiment_ids=["exp1", "exp2"],  # or "exp1" for single
            tags={"environment": "production", "version": "1.0", "mlflow.user": "john_doe"},
        )
        dataset.merge_records(records_df)

    Databricks Usage::

        dataset = create_evaluation_dataset(
            name="catalog.schema.table",
            experiment_ids="exp1",  # or ["exp1", "exp2"]
        )
        dataset.merge_records(records_df)
    """
    if isinstance(experiment_ids, str):
        experiment_ids_list = [experiment_ids]
    else:
        experiment_ids_list = experiment_ids or []

    if is_in_databricks_runtime():
        try:
            from databricks.agents.datasets import create_dataset as db_create

            if tags is not None:
                _logger.warning("Tags are not supported in Databricks environment.")

            return EvaluationDataset(db_create(name, experiment_ids))
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e
    else:
        client = MlflowClient()
        return client.create_evaluation_dataset(
            name=name,
            experiment_ids=experiment_ids_list,
            tags=tags,
        )


@experimental(version="3.3.0")
def get_evaluation_dataset(
    dataset_id: Optional[str] = None, name: Optional[str] = None
) -> EvaluationDataset:
    """
    Get an evaluation dataset by ID (OSS) or name (Databricks).

    Args:
        dataset_id: Dataset ID (required for OSS)
        name: Dataset name/UC table name (required for Databricks)

    OSS Usage::

        dataset = get_evaluation_dataset(dataset_id="dataset_abc123")

    Databricks Usage::

        dataset = get_evaluation_dataset(name="catalog.schema.table")
    """
    if is_in_databricks_runtime():
        if name is None:
            raise ValueError(
                "Parameter 'name' is required in Databricks environment. "
                "Use: get_evaluation_dataset(name='catalog.schema.table')"
            )
        if dataset_id is not None:
            _logger.warning(
                "Parameter 'dataset_id' is ignored in Databricks environment. Use 'name' instead."
            )
        try:
            from databricks.agents.datasets import get_dataset as db_get

            return EvaluationDataset(db_get(name))
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e
    else:
        if dataset_id is None:
            raise ValueError(
                "Parameter 'dataset_id' is required. "
                "Use: get_evaluation_dataset(dataset_id='<dataset_id>')"
            )
        if name is not None:
            _logger.warning("Parameter 'name' is ignored. Use 'dataset_id' instead.")

        client = MlflowClient()
        return client.get_evaluation_dataset(dataset_id)


@experimental(version="3.3.0")
def delete_evaluation_dataset(dataset_id: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Delete an evaluation dataset by ID (OSS) or name (Databricks).

    Args:
        dataset_id: Dataset ID (required for OSS)
        name: Dataset name/UC table name (required for Databricks)

    OSS Usage::

        delete_evaluation_dataset(dataset_id="dataset_abc123")

    Databricks Usage::

        delete_evaluation_dataset(name="catalog.schema.table")
    """
    if is_in_databricks_runtime():
        if name is None:
            raise ValueError(
                "Parameter 'name' is required in Databricks environment. "
                "Use: delete_evaluation_dataset(name='catalog.schema.table')"
            )
        if dataset_id is not None:
            _logger.warning(
                "Parameter 'dataset_id' is ignored in Databricks environment. Use 'name' instead."
            )
        try:
            from databricks.agents.datasets import delete_dataset as db_delete

            return db_delete(name)
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e
    else:
        if dataset_id is None:
            raise ValueError(
                "Parameter 'dataset_id' is required. "
                "Use: delete_evaluation_dataset(dataset_id='<dataset_id>')"
            )
        if name is not None:
            _logger.warning("Parameter 'name' is ignored. Use 'dataset_id' instead.")

        client = MlflowClient()
        client.delete_evaluation_dataset(dataset_id)


@experimental(version="3.3.0")
def search_evaluation_datasets(
    experiment_ids: Optional[Union[str, list[str]]] = None,
    filter_string: Optional[str] = None,
    max_results: int = SEARCH_EVALUATION_DATASETS_MAX_RESULTS,
    order_by: Optional[list[str]] = None,
    page_token: Optional[str] = None,
) -> PagedList[EvaluationDataset]:
    """
    Search for evaluation datasets (OSS only).

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
    if is_in_databricks_runtime():
        raise NotImplementedError(
            "Evaluation Dataset search is not available in Databricks. "
            "Use Unity Catalog search capabilities instead."
        )

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    client = MlflowClient()
    return client.search_evaluation_datasets(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


@deprecated("Use mlflow.genai.datasets.create_evaluation_dataset instead", since="3.3.0")
def create_dataset(
    uc_table_name: str, experiment_id: Optional[Union[str, list[str]]] = None
) -> "EvaluationDataset":
    """
    Create a dataset with the given name and associate it with the given experiment.

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


@deprecated("Use mlflow.genai.datasets.delete_evaluation_dataset instead", since="3.3.0")
def delete_dataset(uc_table_name: str) -> None:
    """
    Delete the dataset with the given name.

    Args:
        uc_table_name: The UC table name of the dataset.
    """
    try:
        from databricks.agents.datasets import delete_dataset
    except ImportError:
        raise ImportError(_ERROR_MSG) from None
    return delete_dataset(uc_table_name)


@deprecated("Use mlflow.genai.datasets.get_evaluation_dataset instead", since="3.3.0")
def get_dataset(uc_table_name: str) -> "EvaluationDataset":
    """
    Get the dataset with the given name.

    Args:
        uc_table_name: The UC table name of the dataset.
    """
    try:
        from databricks.agents.datasets import get_dataset
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return EvaluationDataset(get_dataset(uc_table_name))


@experimental(version="3.3.0")
def set_evaluation_dataset_tags(
    dataset_id: str,
    tags: dict[str, Any],
    updated_by: Optional[str] = None,
) -> None:
    """
    Set tags for an evaluation dataset.

    This implements a batch tag operation - existing tags are merged with new tags.
    To remove a tag, set its value to None.

    Args:
        dataset_id: The ID of the dataset.
        tags: Dictionary of tags to set. Setting a value to None removes the tag.
        updated_by: The user making the update.

    Usage::

        set_evaluation_dataset_tags(
            dataset_id="dataset_abc123",
            tags={
                "environment": "production",
                "version": "2.0",
                "deprecated": None,  # This removes the 'deprecated' tag
            },
        )

    Note:
        This API is not available in Databricks environments yet.
    """
    if is_in_databricks_runtime():
        raise NotImplementedError(
            "Evaluation Dataset tag operations are not available in Databricks yet. "
            "Tags are managed through Unity Catalog."
        )

    if tags is None:
        raise ValueError("'tags' must be provided")

    client = MlflowClient()
    client.set_evaluation_dataset_tags(dataset_id, tags, updated_by)


@experimental(version="3.3.0")
def delete_evaluation_dataset_tag(
    dataset_id: str,
    key: str,
) -> None:
    """
    Delete a tag from an evaluation dataset.

    Args:
        dataset_id: The ID of the dataset.
        key: The tag key to delete.

    Usage::

        delete_evaluation_dataset_tag(dataset_id="dataset_abc123", key="deprecated")

    Note:
        This API is not available in Databricks environments yet.
    """
    if is_in_databricks_runtime():
        raise NotImplementedError(
            "Evaluation Dataset tag operations are not available in Databricks yet. "
            "Tags are managed through Unity Catalog."
        )

    client = MlflowClient()
    client.delete_evaluation_dataset_tag(dataset_id, key)


__all__ = [
    "create_evaluation_dataset",
    "get_evaluation_dataset",
    "delete_evaluation_dataset",
    "search_evaluation_datasets",
    "set_evaluation_dataset_tags",
    "delete_evaluation_dataset_tag",
    "EvaluationDataset",
    # Deprecated APIs
    "create_dataset",
    "delete_dataset",
    "get_dataset",
]
