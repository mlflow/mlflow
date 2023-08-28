from typing import Any, Dict

from mlflow.data.dataset_source import DatasetSource
from mlflow.utils.annotations import experimental


@experimental
class CodeDatasetSource(DatasetSource):
    def __init__(
        self,
        path: str,
        config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[
            Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
        ] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        task: Optional[str] = None,
    ):
        """
        :param path: The path of the code dataset.
        :param config_name: The name of the code dataset configuration.
        :param data_dir: The `data_dir` of the code dataset configuration.
        :param data_files: Paths to source data file(s) for the code dataset configuration.
        :param split: The split of the code dataset.
        :param revision: Version of the code dataset script to load.
        :param task: The task to prepare the code dataset for during training and evaluation.
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
        return "code"

    def load(self, **kwargs):
        """
        Load the code dataset based on the provided parameters.

        :param kwargs: Additional parameters to customize the loading process.
        :return: Path to the saved Parquet file on DBFS.
        """
        # Load from DBFS (Databricks File System)
        dbfs_path = kwargs.get('dbfs_path')
        if dbfs_path:
            with open(dbfs_path, 'r') as file:
                data = file.read()
            return data

        # Download from a URL, convert to DataFrame, and save to DBFS as Parquet
        url = kwargs.get('url')
        dbfs_save_path = kwargs.get('dbfs_save_path', '/dbfs/downloaded_data.parquet')  # default DBFS save path
        if url:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Assuming the data is in CSV format; adjust if different
            data = io.StringIO(response.text)
            df = pd.read_csv(data)
            
            # Save DataFrame as Parquet to DBFS
            df.to_parquet(dbfs_save_path)
            
            return dbfs_save_path

        raise ValueError("Provide either a valid dbfs_path or url to load data.")


    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> "CodeDatasetSource":
        raise NotImplementedError

    def _to_dict(self) -> Dict[Any, Any]:
        return {"tags": self._tags}

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> "CodeDatasetSource":
        return cls(
            tags=source_dict.get("tags"),
        )
