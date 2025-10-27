from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

from mlflow.data.dataset_source import DatasetSource

if TYPE_CHECKING:
    import datasets


class HuggingFaceDatasetSource(DatasetSource):
    """Represents the source of a Hugging Face dataset used in MLflow Tracking."""

    def __init__(
        self,
        path: str,
        config_name: str | None = None,
        data_dir: str | None = None,
        data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        split: Union[str, "datasets.Split"] | None = None,
        revision: Union[str, "datasets.Version"] | None = None,
        trust_remote_code: bool | None = None,
    ):
        """Create a `HuggingFaceDatasetSource` instance.

        Arguments in `__init__` match arguments of the same name in
        `datasets.load_dataset() <https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/loading_methods#datasets.load_dataset>`_.
        The only exception is `config_name` matches `name` in `datasets.load_dataset()`, because
        we need to differentiate from `mlflow.data.Dataset` `name` attribute.

        Args:
            path: The path of the Hugging Face dataset, if it is a dataset from HuggingFace hub,
                `path` must match the hub path, e.g., "databricks/databricks-dolly-15k".
            config_name: The name of of the Hugging Face dataset configuration.
            data_dir: The `data_dir` of the Hugging Face dataset configuration.
            data_files: Paths to source data file(s) for the Hugging Face dataset configuration.
            split: Which split of the data to load.
            revision: Version of the dataset script to load.
            trust_remote_code: Whether to trust remote code from the dataset repo.
        """
        self.path = path
        self.config_name = config_name
        self.data_dir = data_dir
        self.data_files = data_files
        self.split = split
        self.revision = revision
        self.trust_remote_code = trust_remote_code

    @staticmethod
    def _get_source_type() -> str:
        return "hugging_face"

    def load(self, **kwargs):
        """Load the Hugging Face dataset based on `HuggingFaceDatasetSource`.

        Args:
            kwargs: Additional keyword arguments used for loading the dataset with the Hugging Face
                `datasets.load_dataset()` method.

        Returns:
            An instance of `datasets.Dataset`.
        """
        import datasets
        from packaging.version import Version

        load_kwargs = {
            "path": self.path,
            "name": self.config_name,
            "data_dir": self.data_dir,
            "data_files": self.data_files,
            "split": self.split,
            "revision": self.revision,
        }

        # this argument only exists in >= 2.16.0
        if Version(datasets.__version__) >= Version("2.16.0"):
            load_kwargs["trust_remote_code"] = self.trust_remote_code

        intersecting_keys = set(load_kwargs.keys()) & set(kwargs.keys())
        if intersecting_keys:
            raise KeyError(
                f"Found duplicated arguments in `HuggingFaceDatasetSource` and "
                f"`kwargs`: {intersecting_keys}. Please remove them from `kwargs`."
            )
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

    def to_dict(self) -> dict[Any, Any]:
        return {
            "path": self.path,
            "config_name": self.config_name,
            "data_dir": self.data_dir,
            "data_files": self.data_files,
            "split": str(self.split),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> "HuggingFaceDatasetSource":
        return cls(
            path=source_dict.get("path"),
            config_name=source_dict.get("config_name"),
            data_dir=source_dict.get("data_dir"),
            data_files=source_dict.get("data_files"),
            split=source_dict.get("split"),
            revision=source_dict.get("revision"),
        )
