import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_tensor_digest, compute_tensorflow_dataset_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental
class TensorFlowDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a TensorFlow dataset for use with MLflow Tracking.
    """

    def __init__(
        self,
        features,
        source: DatasetSource,
        targets=None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param features: A TensorFlow dataset or tensor of features.
        :param source: The source of the TensorFlow dataset.
        :param targets: A TensorFlow dataset or tensor of targets. Optional.
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                       is automatically computed.
        """
        import tensorflow as tf

        if not isinstance(features, tf.data.Dataset) and not tf.is_tensor(features):
            raise MlflowException(
                f"'features' must be an instance of tf.data.Dataset or a TensorFlow Tensor."
                f" Found: {type(features)}.",
                INVALID_PARAMETER_VALUE,
            )

        if tf.is_tensor(features) and targets is not None and not tf.is_tensor(targets):
            raise MlflowException(
                f"If 'features' is a TensorFlow Tensor, then 'targets' must also be a TensorFlow"
                f" Tensor. Found: {type(targets)}.",
                INVALID_PARAMETER_VALUE,
            )

        if (
            isinstance(features, tf.data.Dataset)
            and targets is not None
            and not isinstance(targets, tf.data.Dataset)
        ):
            raise MlflowException(
                "If 'features' is an instance of tf.data.Dataset, then 'targets' must also be an"
                f" instance of tf.data.Dataset. Found: {type(targets)}.",
                INVALID_PARAMETER_VALUE,
            )

        self._features = features
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        import tensorflow as tf

        return (
            compute_tensorflow_dataset_digest(self._features, self._targets)
            if isinstance(self._features, tf.data.Dataset)
            else compute_tensor_digest(self._features, self._targets)
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
        return {
            **base_dict,
            "schema": json.dumps(self.schema.to_dict()) if self.schema else None,
            "profile": json.dumps(self.profile),
        }

    @property
    def data(self):
        """
        The underlying TensorFlow data.
        """
        return self._features

    @property
    def source(self) -> DatasetSource:
        """
        The source of the dataset.
        """
        return self._source

    @property
    def targets(self):
        """
        The targets of the dataset.
        """
        return self._targets

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        import tensorflow as tf

        profile = {
            "features_cardinality": int(self._features.cardinality().numpy())
            if isinstance(self._features, tf.data.Dataset)
            else int(tf.size(self._features).numpy()),
        }
        if self._targets is not None:
            profile.update(
                {
                    "targets_cardinality": int(self._targets.cardinality().numpy())
                    if isinstance(self._targets, tf.data.Dataset)
                    else int(tf.size(self._targets).numpy()),
                }
            )
        return profile

    @cached_property
    def schema(self) -> Optional[TensorDatasetSchema]:
        """
        An MLflow TensorSpec schema representing the tensor dataset
        """
        try:
            features_schema = TensorFlowDataset._get_tf_object_schema(self._features)
            targets_schema = None
            if self._targets is not None:
                targets_schema = TensorFlowDataset._get_tf_object_schema(self._targets)
            return TensorDatasetSchema(features=features_schema, targets=targets_schema)
        except Exception as e:
            _logger.warning("Failed to infer schema for TensorFlow dataset. Exception: %s", e)
            return None

    @staticmethod
    def _get_tf_object_schema(tf_object) -> Schema:
        import tensorflow as tf

        if isinstance(tf_object, tf.data.Dataset):
            numpy_data = next(tf_object.as_numpy_iterator())
            if isinstance(numpy_data, np.ndarray):
                return _infer_schema(numpy_data)
            elif isinstance(numpy_data, dict):
                return TensorFlowDataset._get_schema_from_tf_dataset_dict_numpy_data(numpy_data)
            elif isinstance(numpy_data, tuple):
                return TensorFlowDataset._get_schema_from_tf_dataset_tuple_numpy_data(numpy_data)
            else:
                raise MlflowException(
                    f"Failed to infer schema for tf.data.Dataset due to unrecognized numpy iterator"
                    f" data type. Numpy iterator data types 'np.ndarray', 'dict', and 'tuple' are"
                    f" supported. Found: {type(numpy_data)}.",
                    INVALID_PARAMETER_VALUE,
                )
        elif tf.is_tensor(tf_object):
            return _infer_schema(tf_object.numpy())
        else:
            raise MlflowException(
                f"Cannot infer schema of an object that is not an instance of tf.data.Dataset or"
                f" a TensorFlow Tensor. Found: {type(tf_object)}",
                INTERNAL_ERROR,
            )

    @staticmethod
    def _get_schema_from_tf_dataset_dict_numpy_data(numpy_data: Dict[Any, Any]) -> Schema:
        if not all(isinstance(data_element, np.ndarray) for data_element in numpy_data.values()):
            raise MlflowException(
                "Failed to infer schema for tf.data.Dataset. Schemas can only be inferred"
                " if the dataset consists of tensors. Ragged tensors, tensor arrays, and"
                " other types are not supported. Additionally, datasets with nested tensors"
                " are not supported.",
                INVALID_PARAMETER_VALUE,
            )
        return _infer_schema(numpy_data)

    @staticmethod
    def _get_schema_from_tf_dataset_tuple_numpy_data(numpy_data: Tuple[Any]) -> Schema:
        if not all(isinstance(data_element, np.ndarray) for data_element in numpy_data):
            raise MlflowException(
                "Failed to infer schema for tf.data.Dataset. Schemas can only be inferred"
                " if the dataset consists of tensors. Ragged tensors, tensor arrays, and"
                " other types are not supported. Additionally, datasets with nested tensors"
                " are not supported.",
                INVALID_PARAMETER_VALUE,
            )
        return _infer_schema(
            {
                # MLflow Schemas currently require each tensor to have a name, if more than
                # one tensor is defined. Accordingly, use the index as the name
                str(i): data_element
                for i, data_element in enumerate(numpy_data)
            }
        )

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        """
        return PyFuncInputsOutputs(self._features, self._targets)

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Only supported if the
        dataset is a Tensor. Required for use with mlflow.evaluate().
        """
        import tensorflow as tf

        # check that data and targets are Tensors
        if not tf.is_tensor(self._features):
            raise MlflowException("Data must be a Tensor to convert to an EvaluationDataset.")
        if self._targets is not None and not tf.is_tensor(self._targets):
            raise MlflowException("Targets must be a Tensor to convert to an EvaluationDataset.")
        return EvaluationDataset(
            data=self._features.numpy(),
            targets=self._targets.numpy() if self._targets is not None else None,
            path=path,
            feature_names=feature_names,
        )


def from_tensorflow(
    features,
    source: Optional[Union[str, DatasetSource]] = None,
    targets=None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> TensorFlowDataset:
    """
    Constructs a TensorFlowDataset object from TensorFlow data, optional targets, and source.
    If the source is path like, then this will construct a DatasetSource object from the source
    path. Otherwise, the source is assumed to be a DatasetSource object.

    :param features: A TensorFlow dataset or tensor of features.
    :param source: The source from which the data was derived, e.g. a filesystem
                    path, an S3 URI, an HTTPS URL, a delta table name with version, or
                    spark table etc. If source is not a path like string,
                    pass in a DatasetSource object directly. If no source is specified,
                    a CodeDatasetSource is used, which will source information from the run
                    context.
    :param targets: A TensorFlow dataset or tensor of targets. Optional.
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: A dataset digest (hash). If unspecified, a digest is computed
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
    return TensorFlowDataset(
        features=features, source=resolved_source, targets=targets, name=name, digest=digest
    )
