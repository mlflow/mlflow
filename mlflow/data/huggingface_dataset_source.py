import warnings

import datasets

from typing import TypeVar, Any, Union, Optional, Mapping, Sequence, Dict

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
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
        config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[
            Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
        ] = None,
        split: Optional[Union[str, datasets.Split]] = None,
        revision: Optional[Union[str, datasets.Version]] = None,
        task: Optional[Union[str, datasets.TaskTemplate]] = None,
    ):
        self._path = path
        self._config_name = config_name
        self._data_dir = data_dir
        self._data_files = data_files
        self._split = split
        self._revision = revision
        self._task = task

    @staticmethod
    def _get_source_type() -> str:
        return "huggingface"

    def load(self, **kwargs) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """
        Loads the dataset source as a Hugging Face Dataset or DatasetDict, depending on whether
        multiple splits are defined by the source or not.

        :param kwargs: Additional keyword arguments used for loading the dataset with
                       the Hugging Face `datasets.load_dataset()` method. The following keyword
                       arguments are used automatically from the dataset source but may be overriden
                       by values passed in **kwargs: path, name, data_dir, data_files, split,
                       revision, task.
        :throws: MlflowException if the Hugging Face dataset source does not define a path
                 from which to load the data.
        :return: An instance of `datasets.Dataset` or `datasets.DatasetDict`, depending on whether
                 multiple splits are defined by the source or not.
        """
        load_kwargs = {
            "path": self._path,
            "name": self._config_name,
            "data_dir": self._data_dir,
            "data_files": self._data_files,
            "split": self._split,
            "revision": self._revision,
            "task": self._task,
        }
        print("LOAD KWARGS", load_kwargs)
        load_kwargs.update(kwargs)

        return datasets.load_dataset(**load_kwargs)

    @staticmethod
    def _can_resolve(raw_source: Any):
        # NB: Initially, we expect that Hugging Face dataset sources will only be used with
        # Hugging Face datasets constructed by from_huggingface_dataset, which can create
        # an instance of HuggingFaceDatasetSource directly without the need for resolution
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> HuggingFaceDatasetSourceType:
        raise NotImplementedError

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "path": self._path,
            "config_name": self._config_name,
            "data_dir": self._data_dir,
            "data_files": self._data_files,
            "split": str(self._split),
            "revision": self._revision,
            "task": self._task,
        }

    @classmethod
    def _from_dict(cls, source_dict: Dict[str, Any]) -> HuggingFaceDatasetSourceType:
        return cls(
            path=source_dict.get("path"),
            config_name=source_dict.get("config_name"),
            data_dir=source_dict.get("data_dir"),
            data_files=source_dict.get("data_files"),
            split=source_dict.get("split"),
            revision=source_dict.get("revision"),
            task=source_dict.get("task"),
        )
