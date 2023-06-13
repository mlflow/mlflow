import json
import logging
from typing import Optional, Any, Dict, Union

import pandas as pd
from functools import cached_property

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_pandas_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


class PandasDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Pandas DataFrame for use with MLflow Tracking.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source: DatasetSource,
        targets: str = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param df: A pandas DataFrame.
        :param source: The source of the pandas DataFrame.
        :param targets: The name of the target column. Optional.
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                       is automatically computed.
        """
        if targets is not None and targets not in df.columns:
            raise MlflowException(
                f"The specified pandas DataFrame does not contain the specified targets column"
                f" '{targets}'.",
                INVALID_PARAMETER_VALUE,
            )
        self._df = df
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        return compute_pandas_digest(self._df)

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """
        return {
            **base_dict,
            "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()})
            if self.schema
            else None,
            "profile": json.dumps(self.profile),
        }

    @property
    def df(self) -> pd.DataFrame:
        """
        The underlying pandas DataFrame.
        """
        return self._df

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
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be ``None`` if a profile cannot be computed.
        """
        return {
            "num_rows": len(self._df),
            "num_elements": int(self._df.size),
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
            inputs = self._df.drop(columns=[self._targets])
            outputs = self._df[self._targets]
            return PyFuncInputsOutputs(inputs, outputs)
        else:
            return PyFuncInputsOutputs(self._df)

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
        )


@experimental
def from_pandas(
    df: pd.DataFrame,
    source: Union[str, DatasetSource] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> PandasDataset:
    """
    Constructs a :py:class:`PandasDataset <mlflow.data.pandas_dataset.PandasDataset>` instance from
    a Pandas DataFrame, optional targets, and source.

    :param df: A Pandas DataFrame.
    :param source: The source from which the DataFrame was derived, e.g. a filesystem
                   path, an S3 URI, an HTTPS URL, a delta table name with version, or
                   spark table etc. ``source`` may be specified as a URI, a path-like string,
                   or an instance of
                   :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
                   If unspecified, the source is assumed to be the code location
                   (e.g. notebook cell, script, etc.) where
                   :py:func:`from_pandas <mlflow.data.from_pandas>` is being called.
    :param targets: An optional target column name for supervised training. This column
                    must be present in the dataframe (``df``).
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: The dataset digest (hash). If unspecified, a digest is computed
                   automatically.
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
    return PandasDataset(df=df, source=resolved_source, targets=targets, name=name, digest=digest)
