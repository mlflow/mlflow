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
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        split: Optional[Union[str, datasets.Split]] = None,
        features: Optional[datasets.Features] = None,
        revision: Optional[Union[str, datasets.Version]] = None,
        task: Optional[Union[str, datasets.TaskTemplate]] = None,
    ):
        self.path = path
        self.name = name
        self.data_dir = data_dir
        self.data_files = data_files
        self.split = split
        self.features = features
        self.revision = revision
        self.task = task

    @staticmethod
    def _get_source_type() -> str:
        return "huggingface"

    def load(self, **kwargs) -> Union[datasets.Dataset, datasets.DatasetDict]:
        return datasets.load_dataset(self.repository, self.split, self.revision)

    @staticmethod
    def _can_resolve(raw_source: Any):
        return isinstance(raw_source, datasets.Dataset)

    @classmethod
    def _resolve(cls, raw_source: str) -> HuggingFaceDatasetSourceType:
        # TODO: Implement this
        raise NotImplementedError

    def _to_dict(self) -> Dict[str, str]:
        # TODO: Implement this
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, source_dict: Dict[str, str]) -> HuggingFaceDatasetSourceType:
        # TODO: Implement this
        raise NotImplementedError
