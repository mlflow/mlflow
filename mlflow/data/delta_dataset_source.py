from typing import TypeVar, Any, Optional, Dict

from mlflow.data.dataset_source import DatasetSource
from pyspark.sql import SparkSession, DataFrame


DeltaDatasetSourceType = TypeVar("DeltaDatasetSourceType", bound="DeltaDatasetSource")


class DeltaDatasetSource(DatasetSource):
    def __init__(
        self,
        path: Optional[str] = None,
        delta_table_name: Optional[str] = None,
        delta_table_version: Optional[str] = None,
    ):
        self._path = path
        self._delta_table_name = delta_table_name
        self._delta_table_version = delta_table_version

    @staticmethod
    def _get_source_type() -> str:
        return "delta_table"

    def load(self, **kwargs) -> DataFrame:
        """
        Loads the dataset source as a Hugging Face Dataset or DatasetDict, depending on whether
        multiple splits are defined by the source or not.
        :param kwargs: Additional keyword arguments used for loading the dataset with
                       the Hugging Face `datasets.load_dataset()` method. The following keyword
                       arguments are used automatically from the dataset source but may be overriden
                       by values passed in **kwargs: path, name, data_dir, data_files, split,
                       revision, task.
        :throws: MlflowException if the Spark dataset source does not define a path
                 from which to load the data.
        :return: An instance of `pyspark.sql.DataFrame`.
        """
        spark = SparkSession.builder.getOrCreate()

        spark_read_op = spark.read.format("delta")
        if self._delta_table_version is not None:
            spark_read_op = spark_read_op.option("versionAsOf", self._delta_table_version)

        # Read the Delta table using spark.read.format and table method
        if self._path:
            return spark_read_op.load(self._path)
        else:
            return spark_read_op.table(self._delta_table_name)

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> DeltaDatasetSourceType:
        raise NotImplementedError

    def _to_dict(self) -> Dict[Any, Any]:
        return {
            "path": self._path,
        }

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> DeltaDatasetSourceType:
        return cls(
            path=source_dict.get("path"),
        )
