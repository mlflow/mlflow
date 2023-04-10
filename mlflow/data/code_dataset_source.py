from typing import TypeVar, Any, Dict

from mlflow.data.dataset_source import DatasetSource


CodeDatasetSourceType = TypeVar("CodeDatasetSourceType", bound="CodeDatasetSource")


class CodeDatasetSource(DatasetSource):
    def __init__(
        self,
        mlflow_source_type: str,
        mlflow_source_name: str,
    ):
        self._mlflow_source_type = mlflow_source_type
        self._mlflow_source_name = mlflow_source_name

    @staticmethod
    def _get_source_type() -> str:
        return "code"

    def load(self, **kwargs):
        """
        Load is not implemented for Code Dataset Source.
        """
        raise NotImplementedError

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> CodeDatasetSourceType:
        raise NotImplementedError

    def _to_dict(self) -> Dict[Any, Any]:
        return {
            "mlflow_source_type": self._mlflow_source_type,
            "mlflow_source_name": self._mlflow_source_name,
        }

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> CodeDatasetSourceType:
        return cls(
            mlflow_source_type=source_dict.get("mlflow_source_type"),
            mlflow_source_name=source_dict.get("mlflow_source_name"),
        )
