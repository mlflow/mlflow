import json
import logging
from typing import Optional, Any, Dict, Union

import pandas as pd
from functools import cached_property

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_tensor_digest, compute_tensorflow_dataset_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema

import tensorflow as tf

_logger = logging.getLogger(__name__)


class TensorflowDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Tensorflow DataFrame for use with MLflow Tracking.
    """

    def __init__(
        self,
        data: Union[tf.Tensor, tf.data.Dataset],
        source: DatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param data: A Tensorflow dataset or Tensorflow tensor.
        :param source: The source of the Tensorflow dataset.
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                       is automatically computed.
        """
        self._data = data
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        return (
            compute_tensorflow_dataset_digest(self._data)
            if isinstance(self._data, tf.data.Dataset)
            else compute_tensor_digest(self._data)
        )

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """
        base_dict.update(
            {
                "schema": json.dumps({"mlflow_tensorspec": self.schema.to_dict()}),
                "profile": json.dumps(self.profile),
            }
        )
        return base_dict

    @property
    def data(self) -> pd.DataFrame:
        """
        The underlying Tensorflow data.
        """
        return self._data

    @property
    def source(self) -> DatasetSource:
        """
        The source of the dataset.
        """
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        return {
            "num_rows": len(self._data),
            "num_elements": int(self._data.cardinality().numpy())
            if isinstance(self._data, tf.data.Dataset)
            else tf.size(self._data).numpy(),
        }

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        An MLflow TensorSpec schema representing the tensor dataset
        """
        try:
            return (
                _infer_schema(next(self._data.as_numpy_iterator()))
                if isinstance(self._data, tf.data.Dataset)
                else _infer_schema(self._data.numpy())
            )
        except Exception as e:
            _logger._warning("Failed to infer schema for Tensorflow dataset. Exception: %s", e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        """
        return PyFuncInputsOutputs(self._data)


def from_tensorflow(
    data: Union[tf.Tensor, tf.data.Dataset],
    source: Union[str, DatasetSource],
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> TensorflowDataset:
    """
    Constructs a TensorflowDataset object from Tensorflow DataFrame, optional targets, and source.
    If the source is path like, then this will construct a DatasetSource object from the source
    path. Otherwise, the source is assumed to be a DatasetSource object.
    :param data: A Tensorflow dataset or Tensorflow tensor.
    :param source: The source from which the DataFrame was derived, e.g. a filesystem
                    path, an S3 URI, an HTTPS URL, a delta table name with version, or
                    spark table etc. If source is not a path like string,
                    pass in a DatasetSource object directly.
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: A dataset digest (hash). If unspecified, a digest is computed
                    automatically.
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    if isinstance(source, DatasetSource):
        resolved_source = source
    else:
        resolved_source = resolve_dataset_source(
            source,
        )
    return TensorflowDataset(data=data, source=resolved_source, name=name, digest=digest)
