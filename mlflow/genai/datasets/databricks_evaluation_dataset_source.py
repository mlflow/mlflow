from typing import Any, Optional

from mlflow.data.dataset_source import DatasetSource


class DatabricksEvaluationDatasetSource(DatasetSource):
    """
    Represents a Databricks Evaluation Dataset source.

    This source is used for datasets managed by the Databricks agents SDK.
    """

    def __init__(self, table_name: Optional[str] = None, dataset_id: Optional[str] = None):
        """
        Args:
            table_name: The UC table name of the dataset
            dataset_id: The unique identifier of the dataset
        """
        if not table_name and not dataset_id:
            raise ValueError("Either table_name or dataset_id must be provided")
        self._table_name = table_name
        self._dataset_id = dataset_id

    @property
    def table_name(self) -> Optional[str]:
        """The UC table name of the dataset."""
        return self._table_name

    @property
    def dataset_id(self) -> Optional[str]:
        """The unique identifier of the dataset."""
        return self._dataset_id

    @staticmethod
    def _get_source_type() -> str:
        return "databricks_evaluation_dataset"

    def load(self, **kwargs) -> Any:
        """
        Loads the dataset from the source.

        This method is not implemented as the dataset should be loaded through
        the databricks.agents.datasets API.
        """
        raise NotImplementedError(
            "DatabricksEvaluationDatasetSource.load() is not implemented. "
            "Please use the databricks.agents.datasets API to load the dataset."
        )

    @staticmethod
    def _can_resolve(raw_source: dict[str, Any]) -> bool:
        """
        Determines whether the source can be resolved from a dictionary representation.
        """
        # Resolution from a dictionary representation is not supported for Databricks Evaluation
        # Datasets
        return False

    @classmethod
    def _resolve(cls, raw_source: dict[str, Any]):
        """
        Resolves the source from a dictionary representation.
        """
        raise NotImplementedError("Resolution from a source dataset is not supported")

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the source.
        """
        result = {}
        if self._table_name is not None:
            result["table_name"] = self._table_name
        if self._dataset_id is not None:
            result["dataset_id"] = self._dataset_id
        return result

    @classmethod
    def from_dict(cls, source_dict: dict[str, Any]) -> "DatabricksEvaluationDatasetSource":
        """
        Creates an instance from a dictionary representation.
        """
        return cls(
            table_name=source_dict.get("table_name"), dataset_id=source_dict.get("dataset_id")
        )
