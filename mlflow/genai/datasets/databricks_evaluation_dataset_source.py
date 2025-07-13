from typing import Any

from mlflow.data.dataset_source import DatasetSource


class DatabricksEvaluationDatasetSource(DatasetSource):
    """
    Represents a Databricks Evaluation Dataset source.

    This source is used for datasets managed by the Databricks agents SDK.
    """

    def __init__(self, table_name: str, dataset_id: str):
        """
        Args:
            table_name: The three-level UC table name of the dataset
            dataset_id: The unique identifier of the dataset
        """
        self._table_name = table_name
        self._dataset_id = dataset_id

    @property
    def table_name(self) -> str:
        """The UC table name of the dataset."""
        return self._table_name

    @property
    def dataset_id(self) -> str:
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
            "Loading a Databricks Evaluation Dataset from source is not supported"
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
        raise NotImplementedError("Resolution from a source dictionary is not supported")

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the source.
        """
        return {
            "table_name": self._table_name,
            "dataset_id": self._dataset_id,
        }

    @classmethod
    def from_dict(cls, source_dict: dict[str, Any]) -> "DatabricksEvaluationDatasetSource":
        """
        Creates an instance from a dictionary representation.
        """
        return cls(table_name=source_dict["table_name"], dataset_id=source_dict["dataset_id"])
