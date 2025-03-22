import json
import logging
from functools import cached_property
from typing import Any, Optional, Union

import narwhals.stable.v1 as nw
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_pandas_digest
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema, ColSpec
from mlflow.types.schema import Array, DataType, Object, Property
from mlflow.types.utils import _infer_schema

_logger = logging.getLogger(__name__)

ColSpecType = Union[DataType, Array, Object, str]

TYPE_MAP = {
    # nw.Binary: DataType.binary,  # TODO(Narwhals): next release this will be available
    nw.Boolean: DataType.boolean,
    nw.Datetime: DataType.datetime,
    nw.Float32: DataType.float,
    nw.Float64: DataType.double,
    nw.Int8: DataType.integer,
    nw.Int16: DataType.integer,
    nw.Int32: DataType.integer,
    nw.Int64: DataType.long,
    nw.String: DataType.string,
}
CLOSE_MAP = {
    nw.Categorical: DataType.string,
    nw.Enum: DataType.string,
    nw.Date: DataType.datetime,
    nw.UInt8: DataType.integer,
    nw.UInt16: DataType.integer,
    nw.UInt32: DataType.long,
    nw.UInt64: DataType.long,
}
# Remaining types:
# nw.Decimal
# nw.Duration
# nw.Time
# nw.Null
# nw.Object
# nw.Unknown

def infer_schema(df: nw.DataFrame) -> Schema:
    return Schema([
        infer_colspec(col_name, col_dtype)
        for col_name, col_dtype in df.collect_schema().items()
    ])

def infer_colspec(
    col_name: str, col_dtype: nw.dtypes.DType, *, allow_unknown: bool = True,
) -> ColSpec:
    return ColSpec(
        type=infer_dtype(dtype=col_dtype, col_name=col_name, allow_unknown=allow_unknown),
        name=col_name,
    )


def infer_dtype(
    dtype: nw.dtypes.DType, col_name: str, *, allow_unknown: bool
) -> ColSpecType:
    dtype_cls = type(dtype)
    mapped = TYPE_MAP.get(dtype_cls)
    if mapped is not None:
        return mapped

    mapped = CLOSE_MAP.get(dtype_cls)
    if mapped is not None:
        logging.warning(
            "Data type of Column '%s' contains dtype=%s which will be mapped to %s."
            " This is not an exact match but is close enough",
            col_name,
            dtype,
            mapped,
        )
        return mapped

    if isinstance(dtype, (nw.Array, nw.List)):
        inner = (
            "Unknown"
            if dtype.inner is None
            else infer_dtype(dtype.inner, f"{col_name}.[]", allow_unknown=allow_unknown)
        )
        return Array(inner)

    if isinstance(dtype, nw.Struct):
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

class PandasDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Pandas DataFrame for use with MLflow Tracking.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source: DatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
        predictions: Optional[str] = None,
    ):
        """
        Args:
            df: A pandas DataFrame.
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
        if targets is not None and targets not in df.columns:
            raise MlflowException(
                f"The specified pandas DataFrame does not contain the specified targets column"
                f" '{targets}'.",
                INVALID_PARAMETER_VALUE,
            )
        if predictions is not None and predictions not in df.columns:
            raise MlflowException(
                f"The specified pandas DataFrame does not contain the specified predictions column"
                f" '{predictions}'.",
                INVALID_PARAMETER_VALUE,
            )
        self._df: nw.DataFrame = nw.from_native(df)
        self._targets = targets
        self._predictions = predictions
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        return compute_pandas_digest(self._df)

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
    def df(self) -> pd.DataFrame:
        """
        The underlying pandas DataFrame.
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
            "num_elements": int(n_rows*n_cols),
        }

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        An instance of :py:class:`mlflow.types.Schema` representing the tabular dataset. May be
        ``None`` if the schema cannot be inferred from the dataset.
        """
        try:
            return infer_schema(self._df)
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
        else:
            return PyFuncInputsOutputs(self._df.to_native())

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return EvaluationDataset(
            data=self._df.to_native(),
            targets=self._targets,
            path=path,
            feature_names=feature_names,
            predictions=self._predictions,
        )


def from_pandas(
    df: pd.DataFrame,
    source: Union[str, DatasetSource] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
    predictions: Optional[str] = None,
) -> PandasDataset:
    """
    Constructs a :py:class:`PandasDataset <mlflow.data.pandas_dataset.PandasDataset>` instance from
    a Pandas DataFrame, optional targets, optional predictions, and source.

    Args:
        df: A Pandas DataFrame.
        source: The source from which the DataFrame was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. ``source`` may be specified as a URI, a path-like string,
            or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
            If unspecified, the source is assumed to be the code location
            (e.g. notebook cell, script, etc.) where
            :py:func:`from_pandas <mlflow.data.from_pandas>` is being called.
        targets: An optional target column name for supervised training. This column
            must be present in the dataframe (``df``).
        name: The name of the dataset. If unspecified, a name is generated.
        digest: The dataset digest (hash). If unspecified, a digest is computed
            automatically.
        predictions: An optional predictions column name for model evaluation. This column
            must be present in the dataframe (``df``).

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        import pandas as pd

        x = pd.DataFrame(
            [["tom", 10, 1, 1], ["nick", 15, 0, 1], ["july", 14, 1, 1]],
            columns=["Name", "Age", "Label", "ModelOutput"],
        )
        dataset = mlflow.data.from_pandas(x, targets="Label", predictions="ModelOutput")
    """

    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    if source is not None:
        if isinstance(source, DatasetSource):
            resolved_source = source
        else:
            resolved_source = resolve_dataset_source(
                source,
            )
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)
    return PandasDataset(
        df=df,
        source=resolved_source,
        targets=targets,
        name=name,
        digest=digest,
        predictions=predictions,
    )
