from typing import Any

from mlflow.data.dataset_source import DatasetSource


class EvaluationDatasetSource(DatasetSource):
    """
    Represents the source of an evaluation dataset stored in MLflow's tracking store.
    """

    def __init__(self, dataset_id: str, version: int | None = None):
        """
        Args:
            dataset_id: The ID of the evaluation dataset.
            version: Immutable dataset revision. If omitted, resolves the current revision.
        """
        self._dataset_id = dataset_id
        self._version = version

    @staticmethod
    def _get_source_type() -> str:
        return "mlflow_evaluation_dataset"

    def load(self) -> Any:
        """
        Loads the evaluation dataset from the tracking store using current tracking URI.

        Returns:
            The EvaluationDataset entity.
        """
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        return store.get_dataset(self._dataset_id, version=self._version)

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        Determines if the raw source is an evaluation dataset ID.
        """
        if isinstance(raw_source, str):
            return raw_source.startswith("d-") and len(raw_source) == 34
        return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> "EvaluationDatasetSource":
        """
        Creates an EvaluationDatasetSource from a dataset ID.
        """
        if not cls._can_resolve(raw_source):
            raise ValueError(f"Cannot resolve {raw_source} as an evaluation dataset ID")

        return cls(dataset_id=raw_source)

    def to_dict(self) -> dict[str, Any]:
        source = {"dataset_id": self._dataset_id}
        if self._version is not None:
            source["version"] = self._version
        return source

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> "EvaluationDatasetSource":
        return cls(
            dataset_id=source_dict["dataset_id"],
            version=source_dict.get("version"),
        )
