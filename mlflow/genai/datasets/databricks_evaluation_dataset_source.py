from typing import Any

from mlflow.data.dataset_source import DatasetSource


class DatabricksEvaluationDatasetSource(DatasetSource):
    """
    Represents a Databricks Evaluation Dataset source.

    This source is used for datasets managed by the Databricks agents SDK.
    """

    def __init__(
        self,
        table_name: str,
        dataset_id: str,
        version: int | None = None,
        alias: str | None = None,
    ):
        """
        Args:
            table_name: The three-level UC table name of the dataset
            dataset_id: The unique identifier of the dataset
            version: The resolved dataset Delta version, if known
            alias: The alias used to resolve the dataset, if any
        """
        self._table_name = table_name
        self._dataset_id = dataset_id
        self._version = version
        self._alias = alias

    @property
    def table_name(self) -> str:
        """The UC table name of the dataset."""
        return self._table_name

    @property
    def dataset_id(self) -> str:
        """The unique identifier of the dataset."""
        return self._dataset_id

    @property
    def version(self) -> int | None:
        """The resolved dataset Delta version."""
        return self._version

    @property
    def alias(self) -> str | None:
        """The alias used to resolve the dataset."""
        return self._alias

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
        result = {
            "table_name": self._table_name,
            "dataset_id": self._dataset_id,
        }
        if self._version is not None:
            result["version"] = self._version
        if self._alias is not None:
            result["alias"] = self._alias
        return result

    @classmethod
    def from_dict(cls, source_dict: dict[str, Any]) -> "DatabricksEvaluationDatasetSource":
        """
        Creates an instance from a dictionary representation.
        """
        return cls(
            table_name=source_dict["table_name"],
            dataset_id=source_dict["dataset_id"],
            version=source_dict.get("version"),
            alias=source_dict.get("alias"),
        )


class DatabricksUCTableDatasetSource(DatabricksEvaluationDatasetSource):
    @staticmethod
    def _get_source_type() -> str:
        return "databricks-uc-table"
