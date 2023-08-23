from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Union

from mlflow.data.dataset_source import DatasetSource
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import datasets


@experimental
class HuggingFaceDatasetSource(DatasetSource):
    """
    Represents the source of a Hugging Face dataset used in MLflow Tracking.
    """

    def __init__(
        self,
        path: str,
        config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[
            Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
        ] = None,
        split: Optional[Union[str, "datasets.Split"]] = None,
        revision: Optional[Union[str, "datasets.Version"]] = None,
        task: Optional[Union[str, "datasets.TaskTemplate"]] = None,
    ):
        """
        :param path: The path of the Hugging Face dataset.
        :param config_name: The name of of the Hugging Face dataset configuration.
        :param data_dir: The `data_dir` of the Hugging Face dataset configuration.
        :param data_files: Paths to source data file(s) for the Hugging Face dataset configuration.
        :param revision: Version of the dataset script to load.
        :param task: The task to prepare the Hugging Face dataset for during training and
                     evaluation.
        """
        self._path = path
        self._config_name = config_name
        self._data_dir = data_dir
        self._data_files = data_files
        self._split = split
        self._revision = revision
        self._task = task

    @staticmethod
    def _get_source_type() -> str:
        return "hugging_face"

    def load(self, **kwargs):
        """
        Loads the dataset source as a Hugging Face Dataset.

        :param kwargs: Additional keyword arguments used for loading the dataset with
                       the Hugging Face ``datasets.load_dataset()`` method. The following keyword
                       arguments are used automatically from the dataset source but may be
                       overridden by values passed in ``**kwargs``: ``path``, ``name``,
                       ``data_dir``, ``data_files``, ``split``, ``revision``, ``task``.
        :return: An instance of ``datasets.Dataset``.
        """
        import datasets

        load_kwargs = {
            "path": self._path,
            "name": self._config_name,
            "data_dir": self._data_dir,
            "data_files": self._data_files,
            "split": self._split,
            "revision": self._revision,
            "task": self._task,
        }
        load_kwargs.update(kwargs)

        return datasets.load_dataset(**load_kwargs)

    @staticmethod
    def _can_resolve(raw_source: Any):
        # NB: Initially, we expect that Hugging Face dataset sources will only be used with
        # Hugging Face datasets constructed by from_huggingface_dataset, which can create
        # an instance of HuggingFaceDatasetSource directly without the need for resolution
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> "HuggingFaceDatasetSource":
        raise NotImplementedError

    def _to_dict(self) -> Dict[Any, Any]:
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
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> "HuggingFaceDatasetSource":
        return cls(
            path=source_dict.get("path"),
            config_name=source_dict.get("config_name"),
            data_dir=source_dict.get("data_dir"),
            data_files=source_dict.get("data_files"),
            split=source_dict.get("split"),
            revision=source_dict.get("revision"),
            task=source_dict.get("task"),
        )
