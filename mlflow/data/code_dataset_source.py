from typing import TypeVar, Any, Dict

from mlflow.data.dataset_source import DatasetSource
from mlflow.utils.annotations import experimental


CodeDatasetSourceType = TypeVar("CodeDatasetSourceType", bound="CodeDatasetSource")


@experimental
class CodeDatasetSource(DatasetSource):
    def __init__(
        self,
        tags: Dict[Any, Any],
    ):
        self._tags = tags

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
        return {"tags": self._tags}

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> CodeDatasetSourceType:
        return cls(
            tags=source_dict.get("tags"),
        )
