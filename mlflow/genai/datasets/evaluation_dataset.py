from typing import TYPE_CHECKING, Optional, Union

from mlflow.data import Dataset

if TYPE_CHECKING:
    import pandas as pd
    import pyspark
    from databricks.agents.datasets import Dataset as ManagedDataset


class EvaluationDataset(Dataset):
    """
    A dataset for storing evaluation records (inputs and expectations).

    Currently, this class is only supported for Databricks managed datasets.
    To use this class, you must have the `databricks-agents` package installed.
    """

    def __init__(self, dataset: "ManagedDataset"):
        self._dataset = dataset

    @property
    def dataset_id(self) -> str:
        """The unique identifier of the dataset."""
        return self._dataset.dataset_id

    @property
    def digest(self) -> Optional[str]:
        """String digest (hash) of the dataset provided by the caller that uniquely identifies"""
        return self._dataset.digest

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
        return self._dataset.source

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

    def insert(
        self,
        records: Union[list[dict], "pd.DataFrame", "pyspark.sql.DataFrame"],
    ) -> "EvaluationDataset":
        """Insert records into the dataset."""
        dataset = self._dataset.insert(records)
        return EvaluationDataset(dataset)

    def to_df(self) -> "pd.DataFrame":
        """Convert the dataset to a pandas DataFrame."""
        return self._dataset.to_df()
