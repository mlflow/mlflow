import json
import logging
from functools import cached_property
from inspect import isclass
from typing import Any, Final, TypedDict

import polars as pl
from polars.datatypes.classes import DataType as PolarsDataType
from polars.datatypes.classes import DataTypeClass as PolarsDataTypeClass

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema

_logger = logging.getLogger(__name__)


def hash_polars_df(df: pl.DataFrame) -> str:
    # probably not the best way to hash, also see:
    # https://github.com/pola-rs/polars/issues/9743
    # https://stackoverflow.com/q/76678160
    return str(df.hash_rows().sum())


ColSpecType = DataType | Array | Object | str
TYPE_MAP: Final[dict[PolarsDataTypeClass, DataType]] = {
    pl.Binary: DataType.binary,
    pl.Boolean: DataType.boolean,
    pl.Datetime: DataType.datetime,
    pl.Float32: DataType.float,
    pl.Float64: DataType.double,
    pl.Int8: DataType.integer,
    pl.Int16: DataType.integer,
    pl.Int32: DataType.integer,
    pl.Int64: DataType.long,
    pl.String: DataType.string,
    pl.Utf8: DataType.string,
}
CLOSE_MAP: Final[dict[PolarsDataTypeClass, DataType]] = {
    pl.Categorical: DataType.string,
    pl.Enum: DataType.string,
    pl.Date: DataType.datetime,
    pl.UInt8: DataType.integer,
    pl.UInt16: DataType.integer,
    pl.UInt32: DataType.long,
}
# Remaining types:
# pl.Decimal
# pl.UInt64
# pl.Duration
# pl.Time
# pl.Null
# pl.Object
# pl.Unknown


def infer_schema(df: pl.DataFrame) -> Schema:
    return Schema([infer_colspec(df[col]) for col in df.columns])


def infer_colspec(col: pl.Series, *, allow_unknown: bool = True) -> ColSpec:
    return ColSpec(
        type=infer_dtype(col.dtype, col.name, allow_unknown=allow_unknown),
        name=col.name,
        required=col.count() > 0,
    )


def infer_dtype(
    dtype: PolarsDataType | PolarsDataTypeClass, col_name: str, *, allow_unknown: bool
) -> ColSpecType:
    cls: PolarsDataTypeClass = dtype if isinstance(dtype, PolarsDataTypeClass) else type(dtype)
    mapped = TYPE_MAP.get(cls)
    if mapped is not None:
        return mapped

    mapped = CLOSE_MAP.get(cls)
    if mapped is not None:
        logging.warning(
            "Data type of Column '%s' contains dtype=%s which will be mapped to %s."
            " This is not an exact match but is close enough",
            col_name,
            dtype,
            mapped,
        )
        return mapped

    if not isinstance(dtype, PolarsDataType):
        return _handle_unknown_dtype(dtype=dtype, col_name=col_name, allow_unknown=allow_unknown)

    if isinstance(dtype, (pl.Array, pl.List)):
        # cannot check inner if not instantiated
        if isclass(dtype):
            if not allow_unknown:
                _raise_unknown_type(dtype)
            return Array("Unknown")

        inner = (
            "Unknown"
            if dtype.inner is None
            else infer_dtype(dtype.inner, f"{col_name}.[]", allow_unknown=allow_unknown)
        )
        return Array(inner)

    if isinstance(dtype, pl.Struct):
        # cannot check fields if not instantiated
        if isclass(dtype):
            if not allow_unknown:
                _raise_unknown_type(dtype)
            return Object([])

        return Object(
            [
                Property(
                    name=field.name,
                    dtype=infer_dtype(
                        field.dtype, f"{col_name}.{field.name}", allow_unknown=allow_unknown
                    ),
                )
                for field in dtype.fields
            ]
        )

    return _handle_unknown_dtype(dtype=dtype, col_name=col_name, allow_unknown=allow_unknown)


def _handle_unknown_dtype(dtype: Any, col_name: str, *, allow_unknown: bool) -> str:
    if not allow_unknown:
        _raise_unknown_type(dtype)

    logging.warning(
        "Data type of Columns '%s' contains dtype=%s, which cannot be mapped to any DataType",
        col_name,
        dtype,
    )
    return str(dtype)


def _raise_unknown_type(dtype: Any) -> None:
    msg = f"Unknown type: {dtype!r}"
    raise ValueError(msg)


class PolarsDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """A polars DataFrame for use with MLflow Tracking."""

    def __init__(
        self,
        df: pl.DataFrame,
        source: DatasetSource,
        targets: str | None = None,
        name: str | None = None,
        digest: str | None = None,
        predictions: str | None = None,
    ) -> None:
        """
        Args:
            df: A polars DataFrame.
            source: Source of the DataFrame.
            targets: Name of the target column. Optional.
            name: Name of the dataset. E.g. "wiki_train". If unspecified, a name is automatically
                generated.
            digest: Digest (hash, fingerprint) of the dataset. If unspecified, a digest is
                automatically computed.
            predictions: Name of the column containing model predictions, if the dataset contains
                model predictions. Optional. If specified, this column must be present in ``df``.
        """
        if targets is not None and targets not in df.columns:
            raise MlflowException(
                f"DataFrame does not contain specified targets column: '{targets}'",
                INVALID_PARAMETER_VALUE,
            )
        if predictions is not None and predictions not in df.columns:
            raise MlflowException(
                f"DataFrame does not contain specified predictions column: '{predictions}'",
                INVALID_PARAMETER_VALUE,
            )

        # _df needs to be set before super init, as it is used in _compute_digest
        # see Dataset.__init__()
        self._df = df
        super().__init__(source=source, name=name, digest=digest)
        self._targets = targets
        self._predictions = predictions

    def _compute_digest(self) -> str:
        """Compute a digest for the dataset.

        Called if the user doesn't supply a digest when constructing the dataset.
        """
        return hash_polars_df(self._df)

    class PolarsDatasetConfig(TypedDict):
        name: str
        digest: str
        source: str
        source_type: str
        schema: str
        profile: str

    def to_dict(self) -> PolarsDatasetConfig:
        """Create config dictionary for the dataset.

        Return a string dictionary containing the following fields: name, digest, source,
        source type, schema, and profile.
        """
        schema = json.dumps({"mlflow_colspec": self.schema.to_dict()} if self.schema else None)
        return {
            "name": self.name,
            "digest": self.digest,
            "source": self.source.to_json(),
            "source_type": self.source._get_source_type(),
            "schema": schema,
            "profile": json.dumps(self.profile),
        }

    @property
    def df(self) -> pl.DataFrame:
        """Underlying DataFrame."""
        return self._df

    @property
    def source(self) -> DatasetSource:
        """Source of the dataset."""
        return self._source

    @property
    def targets(self) -> str | None:
        """Name of the target column.

        May be ``None`` if no target column is available.
        """
        return self._targets

    @property
    def predictions(self) -> str | None:
        """Name of the predictions column.

        May be ``None`` if no predictions column is available.
        """
        return self._predictions

    class PolarsDatasetProfile(TypedDict):
        num_rows: int
        num_elements: int

    @property
    def profile(self) -> PolarsDatasetProfile:
        """Profile of the dataset."""
        return {
            "num_rows": self._df.height,
            "num_elements": self._df.height * self._df.width,
        }

    @cached_property
    def schema(self) -> Schema | None:
        """Instance of :py:class:`mlflow.types.Schema` representing the tabular dataset.

        May be ``None`` if the schema cannot be inferred from the dataset.
        """
        try:
            return infer_schema(self._df)
        except Exception as e:
            _logger.warning("Failed to infer schema for PolarsDataset. Exception: %s", e)
        return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """Convert dataset to a collection of pyfunc inputs and outputs for model evaluation."""
        if self._targets:
            inputs = self._df.drop(*self._targets)
            outputs = self._df.select(self._targets).to_series()
            return PyFuncInputsOutputs([inputs.to_pandas()], [outputs.to_pandas()])
        else:
            return PyFuncInputsOutputs([self._df.to_pandas()])

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """Convert dataset to an EvaluationDataset for model evaluation."""
        return EvaluationDataset(
            data=self._df.to_pandas(),
            targets=self._targets,
            path=path,
            feature_names=feature_names,
            predictions=self._predictions,
        )


def from_polars(
    df: pl.DataFrame,
    source: str | DatasetSource | None = None,
    targets: str | None = None,
    name: str | None = None,
    digest: str | None = None,
    predictions: str | None = None,
) -> PolarsDataset:
    """Construct a :py:class:`PolarsDataset <mlflow.data.polars_dataset.PolarsDataset>` instance.

    Args:
        df: A polars DataFrame.
        source: Source from which the DataFrame was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. ``source`` may be specified as a URI, a path-like string,
            or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
            If unspecified, the source is assumed to be the code location
            (e.g. notebook cell, script, etc.) where
            :py:func:`from_polars <mlflow.data.from_polars>` is being called.
        targets: An optional target column name for supervised training. This column
            must be present in ``df``.
        name: Name of the dataset. If unspecified, a name is generated.
        digest: Dataset digest (hash). If unspecified, a digest is computed
            automatically.
        predictions: An optional predictions column name for model evaluation. This column
            must be present in ``df``.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        import polars as pl

        x = pl.DataFrame(
            [["tom", 10, 1, 1], ["nick", 15, 0, 1], ["julie", 14, 1, 1]],
            schema=["Name", "Age", "Label", "ModelOutput"],
        )
        dataset = mlflow.data.from_polars(x, targets="Label", predictions="ModelOutput")
    """

    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    if source is not None:
        if isinstance(source, DatasetSource):
            resolved_source = source
        else:
            resolved_source = resolve_dataset_source(source)
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)
    return PolarsDataset(
        df=df,
        source=resolved_source,
        targets=targets,
        name=name,
        digest=digest,
        predictions=predictions,
    )
