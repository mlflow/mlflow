from typing import TYPE_CHECKING, Optional, Union

from mlflow.data import Dataset
from mlflow.data.digest_utils import compute_pandas_digest
from mlflow.data.evaluation_dataset import EvaluationDataset as LegacyEvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.entities import Dataset as DatasetEntity

if TYPE_CHECKING:
    import pandas as pd
    import pyspark
    from databricks.agents.datasets import Dataset as ManagedDataset


class EvaluationDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    A dataset for storing evaluation records (inputs and expectations).

    Currently, this class is only supported for Databricks managed datasets.
    To use this class, you must have the `databricks-agents` package installed.
    """

    def __init__(self, dataset: "ManagedDataset"):
        self._dataset = dataset
        self._df = None
        self._digest = None

    @property
    def dataset_id(self) -> str:
        """The unique identifier of the dataset."""
        return self._dataset.dataset_id

    @property
    def digest(self) -> Optional[str]:
        """String digest (hash) of the dataset provided by the caller that uniquely identifies"""
        # NB: The managed Dataset entity in Agent SDK doesn't propagate the digest
        # information. So we compute the digest of the dataframe view.
        if self._digest is None:
            self._digest = self._dataset.digest or compute_pandas_digest(self.to_df())
        return self._digest

    @property
    def name(self) -> Optional[str]:
        """The UC table name of the dataset."""
        return self._dataset.name

    @property
    def schema(self) -> Optional[str]:
        """The schema of the dataset."""
        return self._dataset.schema

    @property
    def profile(self) -> Optional[str]:
        """The profile of the dataset, summary statistics."""
        return self._dataset.profile

    @property
    def source(self) -> Optional[str]:
        """Source information for the dataset."""
        # NB: The managed Dataset entity in Agent SDK doesn't propagate the source
        # information. So we use the table name as the fallback source.
        return self._dataset.source or SparkDatasetSource(table_name=self.name)

    @property
    def source_type(self) -> Optional[str]:
        """The type of the dataset source, e.g. "databricks-uc-table", "DBFS", "S3", ..."""
        return self._dataset.source_type

    @property
    def create_time(self) -> Optional[str]:
        """The time the dataset was created."""
        return self._dataset.create_time

    @property
    def created_by(self) -> Optional[str]:
        """The user who created the dataset."""
        return self._dataset.created_by

    @property
    def last_update_time(self) -> Optional[str]:
        """The time the dataset was last updated."""
        return self._dataset.last_update_time

    @property
    def last_updated_by(self) -> Optional[str]:
        """The user who last updated the dataset."""
        return self._dataset.last_updated_by

    def set_profile(self, profile: str) -> "EvaluationDataset":
        """Set the profile of the dataset."""
        dataset = self._dataset.set_profile(profile)
        return EvaluationDataset(dataset)

    def merge_records(
        self,
        records: Union[list[dict], "pd.DataFrame", "pyspark.sql.DataFrame"],
    ) -> "EvaluationDataset":
        """Merge records into the dataset."""
        dataset = self._dataset.merge_records(records)
        return EvaluationDataset(dataset)

    def to_df(self) -> "pd.DataFrame":
        """Convert the dataset to a pandas DataFrame."""
        # Cache the dataframe view to avoid re-fetching the records many times
        if self._df is None:
            self._df = self._dataset.to_df()
        return self._df

    def to_evaluation_dataset(self, path=None, feature_names=None) -> LegacyEvaluationDataset:
        """
        Converts the dataset to the legacy EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return LegacyEvaluationDataset(
            data=self.to_df(),
            path=path,
            feature_names=feature_names,
            name=self.name,
            digest=self.digest,
        )

    def _to_mlflow_entity(self) -> DatasetEntity:
        return DatasetEntity(
            name=self.name,
            digest=self.digest,
            source_type=self.source_type,
            source=self.source.to_json(),
            schema=self.schema,
            profile=self.profile,
        )
