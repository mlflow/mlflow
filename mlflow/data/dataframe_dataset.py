from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import Any, Generic, Optional, Union

import narwhals.stable.v1 as nw
from narwhals.typing import IntoDataFrameT

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import (
    compute_pandas_digest,
    compute_polars_digest,
    compute_pyarrow_digest,
)
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema

_logger = logging.getLogger(__name__)


class DataFrameDataset(Dataset, PyFuncConvertibleDatasetMixin, Generic[IntoDataFrameT]):
    """
    Represents a (eager) DataFrame for use with MLflow Tracking.
    This class is a generic class that can be instantiated with a pandas, polars or pyarrow.
    It can be used directly, or subclassed to create custom dataset classes.
    """

    def __init__(
        self,
        df: IntoDataFrameT,
        source: DatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
        predictions: Optional[str] = None,
    ) -> None:
        """
        Args:
            df: A eager DataFrame (such as pandas, polars or pyarrow).
            source: The source of the pandas DataFrame.
            targets: The name of the target column. Optional.
            name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                automatically generated.
            digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                is automatically computed.
            predictions: Optional. The name of the column containing model predictions,
                if the dataset contains model predictions. If specified, this column
                must be present in the dataframe (``df``).
        """
        nw_frame = nw.from_native(df, eager_only=True, pass_through=False)
        backend_name = nw_frame.implementation.name.lower()

        if targets is not None and targets not in nw_frame.columns:
            raise MlflowException(
                f"The specified {backend_name} DataFrame does not contain the specified "
                f"targets column '{targets}'.",
                INVALID_PARAMETER_VALUE,
            )
        if predictions is not None and predictions not in nw_frame.columns:
            raise MlflowException(
                f"The specified {backend_name} DataFrame does not contain the specified "
                f"predictions column '{predictions}'.",
                INVALID_PARAMETER_VALUE,
            )
        self._df = nw_frame
        self._targets = targets
        self._predictions = predictions
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        frame_implementation = self._df.implementation
        compute_digest_function = (
            compute_pandas_digest
            if frame_implementation.is_pandas()
            else compute_polars_digest
            if frame_implementation.is_polars()
            else compute_pyarrow_digest
            if frame_implementation.is_pyarrow()
            else None
        )
        return compute_digest_function(self._df.to_native())

    def to_dict(self) -> dict[str, str]:
        """Create config dictionary for the dataset.
        Returns a string dictionary containing the following fields: name, digest, source, source
        type, schema, and profile.
        """
        schema = json.dumps({"mlflow_colspec": self.schema.to_dict()}) if self.schema else None
        config = super().to_dict()
        config.update(
            {
                "schema": schema,
                "profile": json.dumps(self.profile),
            }
        )
        return config

    @property
    def df(self) -> IntoDataFrameT:
        """
        The underlying eager DataFrame.
        """
        return self._df.to_native()

    @property
    def source(self) -> DatasetSource:
        """
        The source of the dataset.
        """
        return self._source

    @property
    def targets(self) -> Optional[str]:
        """
        The name of the target column. May be ``None`` if no target column is available.
        """
        return self._targets

    @property
    def predictions(self) -> Optional[str]:
        """
        The name of the predictions column. May be ``None`` if no predictions column is available.
        """
        return self._predictions

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be ``None`` if a profile cannot be computed.
        """
        n_rows, n_cols = self._df.shape
        return {
            "num_rows": n_rows,
            "num_elements": n_rows * n_cols,
        }

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        An instance of :py:class:`mlflow.types.Schema` representing the tabular dataset. May be
        ``None`` if the schema cannot be inferred from the dataset.
        """
        try:
            return _infer_schema(self._df)
        except Exception as e:
            _logger.warning("Failed to infer schema for Pandas dataset. Exception: %s", e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        """
        if self._targets:
            inputs = self._df.drop(self._targets).to_native()
            outputs = self._df.get_column(self._targets).to_native()
            return PyFuncInputsOutputs(inputs, outputs)

        return PyFuncInputsOutputs(self._df.to_native())

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return EvaluationDataset(
            data=self._df,
            targets=self._targets,
            path=path,
            feature_names=feature_names,
            predictions=self._predictions,
            name=self.name,
            digest=self.digest,
        )


def from_dataframe(
    df: IntoDataFrameT,
    source: Union[str, DatasetSource] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
    predictions: Optional[str] = None,
    dataset_cls: type[DataFrameDataset[IntoDataFrameT]] = DataFrameDataset,
) -> DataFrameDataset[IntoDataFrameT]:
    """
    Constructs a :py:class:`DataFrameDataset <mlflow.data.dataframe_dataset.DataFrameDataset>`
    instance from an eager DataFrame, optional targets, optional predictions, and source.

    Args:
        df: An eager DataFrame (such as pandas, polars or pyarrow).
        source: The source from which the DataFrame was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. ``source`` may be specified as a URI, a path-like string,
            or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
            If unspecified, the source is assumed to be the code location
            (e.g. notebook cell, script, etc.) where
            :py:func:`from_dataframe <mlflow.data.from_dataframe>` is being called.
        targets: An optional target column name for supervised training. This column
            must be present in the dataframe (``df``).
        name: The name of the dataset. If unspecified, a name is generated.
        digest: The dataset digest (hash). If unspecified, a digest is computed
            automatically.
        predictions: An optional predictions column name for model evaluation. This column
            must be present in the dataframe (``df``).
        dataset_cls: The class to use for the dataset. This is useful for creating
            custom dataset classes that inherit from :py:class:`DataFrameDataset
            <mlflow.data.dataframe_dataset.DataFrameDataset>`.
            The default is :py:class:`DataFrameDataset
            <mlflow.data.dataframe_dataset.DataFrameDataset>`.

    .. code-block:: python
        :test:
        :caption: Example
        import mlflow
        import pandas as pd
        import polars as pl

        data = {
            "Name": ["tom", "nick", "july"],
            "Age": [10, 15, 14],
            "Label": [1, 0, 1],
            "ModelOutput": [1, 1, 1],
        }
        df_pd = pd.DataFrame(data)
        dataset_pd = mlflow.data.from_dataframe(x, targets="Label", predictions="ModelOutput")
        df_pl = pl.DataFrame(data)
        dataset_pl = mlflow.data.from_dataframe(x, targets="Label", predictions="ModelOutput")
    """

    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    if source is not None:
        resolved_source = (
            source if isinstance(source, DatasetSource) else resolve_dataset_source(source)
        )
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)

    return dataset_cls(
        df=df,
        source=resolved_source,
        targets=targets,
        name=name,
        digest=digest,
        predictions=predictions,
    )
