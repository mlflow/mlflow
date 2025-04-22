from typing import Any

from typing_extensions import Self

from mlflow.data.dataset_source import DatasetSource


class CodeDatasetSource(DatasetSource):
    def __init__(
        self,
        tags: dict[Any, Any],
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
    def _resolve(cls, raw_source: str) -> Self:
        raise NotImplementedError

    def to_dict(self) -> dict[Any, Any]:
        return {"tags": self._tags}

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> Self:
        return cls(
            tags=source_dict.get("tags"),
        )
