import warnings

import datasets

from typing import TypeVar, Any, Union, Optional, Mapping, Sequence, Dict

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import _artifact_repository_registry

from mlflow.data.dataset_source import DatasetSource


HuggingFaceDatasetSourceType = TypeVar(
    "HuggingFaceDatasetSourceType", bound="HuggingFaceDatasetSource"
)


class HuggingFaceDatasetSource(DatasetSource):
    def __init__(
        self,
        path: str,
        split: Union[str, datasets.splits.Split, None] = None,
        revision: Union[str, datasets.utils.version.Version, None] = None,
        data_files: Union[
            str, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None
        ] = None,
    ):
        self.path = path
        self.split = split
        self.revision = revision
        self.data_files = data_files

    @staticmethod
    def _get_source_type() -> str:
        return "huggingface"

    def download(self) -> str:
        return datasets.load_dataset(self.repository, self.split, self.revision)

    @staticmethod
    def _can_resolve(raw_source: Any):
        return isinstance(raw_source, datasets.Dataset)

    @classmethod
    def _resolve(cls, raw_source: str) -> HuggingFaceDatasetSourceType:
        # TODO: Implement this
        raise NotImplementedError

    def to_dict(self) -> Dict[str, str]:
        # TODO: Implement this
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, source_dict: Dict[str, str]) -> HuggingFaceDatasetSourceType:
        # TODO: Implement this
        raise NotImplementedError
