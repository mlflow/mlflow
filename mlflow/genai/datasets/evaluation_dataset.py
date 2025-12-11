from typing import TYPE_CHECKING, Any

from mlflow.data import Dataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin
from mlflow.entities.evaluation_dataset import EvaluationDataset as _EntityEvaluationDataset
from mlflow.genai.datasets.databricks_evaluation_dataset_source import (
    DatabricksEvaluationDatasetSource,
)

if TYPE_CHECKING:
    import pandas as pd
    import pyspark.sql


class EvaluationDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    The public API for evaluation datasets in MLflow's GenAI module.

    This class provides a unified interface for evaluation datasets, supporting both:

    - Standard MLflow evaluation datasets (backed by MLflow's tracking store)
    - Databricks managed datasets (backed by Unity Catalog tables) through the
      databricks-agents library
    """

    def __init__(self, dataset):
        """
        Initialize the wrapper with either a managed dataset or an MLflow dataset.

        Args:
            dataset: Either a Databricks managed dataset (databricks.agents.datasets.Dataset)
                or an MLflow EvaluationDataset entity
                (mlflow.entities.evaluation_dataset.EvaluationDataset).
                The type is determined at runtime.
        """
        if isinstance(dataset, _EntityEvaluationDataset):
            self._databricks_dataset = None
            self._mlflow_dataset = dataset
        else:
            self._databricks_dataset = dataset
            self._mlflow_dataset = None
        self._df = None

    def __eq__(self, other):
        """Check equality with another dataset."""
        if isinstance(other, _EntityEvaluationDataset) and self._mlflow_dataset:
            return self._mlflow_dataset == other
        if isinstance(other, EvaluationDataset):
            if self._mlflow_dataset and other._mlflow_dataset:
                return self._mlflow_dataset == other._mlflow_dataset
            if self._databricks_dataset and other._databricks_dataset:
                return self._databricks_dataset == other._databricks_dataset
        return False

    def __setattr__(self, name, value):
        """Allow setting internal attributes on the wrapped dataset."""
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        Dynamic attribute delegation for simple pass-through properties.

        This handles attributes that don't require special logic and can be
        directly delegated to the underlying dataset implementation.
        """
        if name.startswith("_") or name == "records":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if self._mlflow_dataset and hasattr(self._mlflow_dataset, name):
            return getattr(self._mlflow_dataset, name)
        elif self._databricks_dataset and hasattr(self._databricks_dataset, name):
            return getattr(self._databricks_dataset, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def digest(self) -> str | None:
        """String digest (hash) of the dataset provided by the caller that uniquely identifies"""
        if self._mlflow_dataset:
            return self._mlflow_dataset.digest
        if self._databricks_dataset.digest is None:
            from mlflow.data.digest_utils import compute_pandas_digest

            return compute_pandas_digest(self.to_df())
        return self._databricks_dataset.digest

    @property
    def name(self) -> str:
        """The name of the dataset."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.name
        return self._databricks_dataset.name if self._databricks_dataset else None

    @property
    def dataset_id(self) -> str:
        """The unique identifier of the dataset."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.dataset_id
        return self._databricks_dataset.dataset_id if self._databricks_dataset else None

    @property
    def source(self):
        """Source information for the dataset."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.source
        return DatabricksEvaluationDatasetSource(table_name=self.name, dataset_id=self.dataset_id)

    @property
    def source_type(self) -> str | None:
        """The type of the dataset source."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.source._get_source_type()
        return self._databricks_dataset.source_type

    @property
    def created_time(self) -> int | str | None:
        """The time the dataset was created."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.created_time
        return self._databricks_dataset.create_time

    @property
    def create_time(self) -> int | str | None:
        """Alias for created_time (for backward compatibility with managed datasets)."""
        return self.created_time

    @property
    def tags(self) -> dict[str, Any] | None:
        """The tags for the dataset (MLflow only)."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.tags
        raise NotImplementedError(
            "Tags are not available for Databricks managed datasets. "
            "Tags are managed through Unity Catalog. Use Unity Catalog APIs to manage dataset tags."
        )

    @property
    def experiment_ids(self) -> list[str]:
        """The experiment IDs associated with the dataset (MLflow only)."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.experiment_ids
        raise NotImplementedError(
            "Experiment associations are not available for Databricks managed datasets. "
            "Dataset associations are managed through Unity Catalog."
        )

    @property
    def schema(self) -> str | None:
        """The schema of the dataset."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.schema
        return self._databricks_dataset.schema if self._databricks_dataset else None

    @property
    def profile(self) -> str | None:
        """The profile of the dataset."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.profile
        return self._databricks_dataset.profile if self._databricks_dataset else None

    def set_profile(self, profile: str) -> "EvaluationDataset":
        """Set the profile of the dataset."""
        if self._mlflow_dataset:
            self._mlflow_dataset._profile = profile
            return self
        dataset = self._databricks_dataset.set_profile(profile)
        return EvaluationDataset(dataset)

    def merge_records(
        self,
        records: "list[dict[str, Any]] | pd.DataFrame | pyspark.sql.DataFrame",
    ) -> "EvaluationDataset":
        """Merge records into the dataset."""
        if self._mlflow_dataset:
            self._mlflow_dataset.merge_records(records)
            return self

        from mlflow.genai.datasets import _databricks_profile_env

        with _databricks_profile_env():
            dataset = self._databricks_dataset.merge_records(records)
        return EvaluationDataset(dataset)

    def to_df(self) -> "pd.DataFrame":
        """Convert the dataset to a pandas DataFrame."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.to_df()

        if self._df is None:
            from mlflow.genai.datasets import _databricks_profile_env

            with _databricks_profile_env():
                self._df = self._databricks_dataset.to_df()
        return self._df

    def has_records(self) -> bool:
        """Check if dataset records are loaded without triggering a load."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.has_records()
        return self._df is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.to_dict()
        raise NotImplementedError(
            "Serialization to dict is not supported for Databricks managed datasets. "
            "Databricks datasets are persisted in Unity Catalog tables and don't "
            "require serialization."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationDataset":
        """
        Create instance from dictionary representation.

        Note: This creates an MLflow dataset from serialized data.
        Databricks managed datasets are loaded directly from Unity Catalog, not from dict.
        """
        mlflow_dataset = _EntityEvaluationDataset.from_dict(data)
        return cls(mlflow_dataset)

    def to_proto(self):
        """Convert to protobuf representation."""
        if self._mlflow_dataset:
            return self._mlflow_dataset.to_proto()
        raise NotImplementedError(
            "Protobuf serialization is not supported for Databricks managed datasets. "
            "Databricks datasets are persisted in Unity Catalog tables and don't "
            "require serialization."
        )

    @classmethod
    def from_proto(cls, proto):
        """
        Create instance from protobuf representation.

        Note: This creates an MLflow dataset from serialized protobuf data.
        Databricks managed datasets are loaded directly from Unity Catalog, not from protobuf.
        """
        mlflow_dataset = _EntityEvaluationDataset.from_proto(proto)
        return cls(mlflow_dataset)

    def _to_pyfunc_dataset(self):
        """Support for PyFuncConvertibleDatasetMixin."""
        return self.to_evaluation_dataset()

    def to_evaluation_dataset(self, path=None, feature_names=None):
        """
        Converts the dataset to the legacy EvaluationDataset for model evaluation.
        Required for use with mlflow.evaluate().
        """
        from mlflow.data.evaluation_dataset import EvaluationDataset as LegacyEvaluationDataset

        return LegacyEvaluationDataset(
            data=self.to_df(),
            path=path,
            feature_names=feature_names,
            name=self.name,
            digest=self.digest,
        )

    def _to_mlflow_entity(self):
        """Convert to MLflow Dataset entity for logging."""
        from mlflow.entities import Dataset as DatasetEntity

        return DatasetEntity(
            name=self.name,
            digest=self.digest,
            source_type=self.source_type,
            source=self.source.to_json(),
            schema=self.schema,
            profile=self.profile,
        )
