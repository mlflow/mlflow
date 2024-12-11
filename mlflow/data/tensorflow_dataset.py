import json
import logging
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import (
    MAX_ROWS,
    compute_numpy_digest,
    get_normalized_md5_digest,
)
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema

_logger = logging.getLogger(__name__)


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
        Args:
            features: A TensorFlow dataset or tensor of features.
            source: The source of the TensorFlow dataset.
            targets: A TensorFlow dataset or tensor of targets. Optional.
            name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                automatically generated.
            digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
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

    def _compute_tensorflow_dataset_digest(  # noqa: D417
        self,
        dataset,
        targets=None,
    ) -> str:
        """Computes a digest for the given Tensorflow dataset.

        Args:
            dataset: A Tensorflow dataset.

        Returns:
            A string digest.
        """
        import pandas as pd
        import tensorflow as tf

        hashable_elements = []

        def hash_tf_dataset_iterator_element(element):
            if element is None:
                return
            flat_element = tf.nest.flatten(element)
            flattened_array = np.concatenate([x.flatten() for x in flat_element])
            trimmed_array = flattened_array[0:MAX_ROWS]
            try:
                hashable_elements.append(pd.util.hash_array(trimmed_array))
            except TypeError:
                hashable_elements.append(np.int64(trimmed_array.size))

        for element in dataset.as_numpy_iterator():
            hash_tf_dataset_iterator_element(element)
        if targets is not None:
            for element in targets.as_numpy_iterator():
                hash_tf_dataset_iterator_element(element)

        return get_normalized_md5_digest(hashable_elements)

    def _compute_tensor_digest(
        self,
        tensor_data,
        tensor_targets,
    ) -> str:
        """Computes a digest for the given Tensorflow tensor.

        Args:
            tensor_data: A Tensorflow tensor, representing the features.
            tensor_targets: A Tensorflow tensor, representing the targets. Optional.

        Returns:
            A string digest.
        """
        if tensor_targets is None:
            return compute_numpy_digest(tensor_data.numpy())
        else:
            return compute_numpy_digest(tensor_data.numpy(), tensor_targets.numpy())

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        import tensorflow as tf

        if isinstance(self._features, tf.data.Dataset):
            return self._compute_tensorflow_dataset_digest(self._features, self._targets)
        return self._compute_tensor_digest(self._features, self._targets)

    def to_dict(self) -> dict[str, str]:
        """Create config dictionary for the dataset.

        Returns a string dictionary containing the following fields: name, digest, source, source
        type, schema, and profile.
        """
        schema = json.dumps(self.schema.to_dict()) if self.schema else None
        config = super().to_dict()
        config.update(
            {
                "schema": schema,
                "profile": json.dumps(self.profile),
            }
        )
        return config

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
    def _get_schema_from_tf_dataset_dict_numpy_data(numpy_data: dict[Any, Any]) -> Schema:
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
    def _get_schema_from_tf_dataset_tuple_numpy_data(numpy_data: tuple[Any]) -> Schema:
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
    """Constructs a TensorFlowDataset object from TensorFlow data, optional targets, and source.

    If the source is path like, then this will construct a DatasetSource object from the source
    path. Otherwise, the source is assumed to be a DatasetSource object.

    Args:
        features: A TensorFlow dataset or tensor of features.
        source: The source from which the data was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. If source is not a path like string,
            pass in a DatasetSource object directly. If no source is specified,
            a CodeDatasetSource is used, which will source information from the run
            context.
        targets: A TensorFlow dataset or tensor of targets. Optional.
        name: The name of the dataset. If unspecified, a name is generated.
        digest: A dataset digest (hash). If unspecified, a digest is computed
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
