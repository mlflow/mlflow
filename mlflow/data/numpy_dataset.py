import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Union

import numpy as np

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_numpy_digest
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.types.utils import _infer_schema

_logger = logging.getLogger(__name__)


class NumpyDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a NumPy dataset for use with MLflow Tracking.
    """

    def __init__(
        self,
        features: Union[np.ndarray, Dict[str, np.ndarray]],
        source: DatasetSource,
        targets: Union[np.ndarray, Dict[str, np.ndarray]] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        Args:
            features: A numpy array or dictionary of numpy arrays containing dataset features.
            source: The source of the numpy dataset.
            targets: A numpy array or dictionary of numpy arrays containing dataset targets.
                Optional.
            name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                automatically generated.
            digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                is automatically computed.
        """
        self._features = features
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        return compute_numpy_digest(self._features, self._targets)

    def to_dict(self) -> Dict[str, str]:
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
    def source(self) -> DatasetSource:
        """
        The source of the dataset.
        """
        return self._source

    @property
    def features(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        The features of the dataset.
        """
        return self._features

    @property
    def targets(self) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        The targets of the dataset. May be ``None`` if no targets are available.
        """
        return self._targets

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be ``None`` if a profile cannot be computed.
        """

        def get_profile_attribute(numpy_data, attr_name):
            if isinstance(numpy_data, dict):
                return {key: getattr(array, attr_name) for key, array in numpy_data.items()}
            else:
                return getattr(numpy_data, attr_name)

        profile = {
            "features_shape": get_profile_attribute(self._features, "shape"),
            "features_size": get_profile_attribute(self._features, "size"),
            "features_nbytes": get_profile_attribute(self._features, "nbytes"),
        }
        if self._targets is not None:
            profile.update(
                {
                    "targets_shape": get_profile_attribute(self._targets, "shape"),
                    "targets_size": get_profile_attribute(self._targets, "size"),
                    "targets_nbytes": get_profile_attribute(self._targets, "nbytes"),
                }
            )

        return profile

    @cached_property
    def schema(self) -> Optional[TensorDatasetSchema]:
        """
        MLflow TensorSpec schema representing the dataset features and targets (optional).
        """
        try:
            features_schema = _infer_schema(self._features)
            targets_schema = None
            if self._targets is not None:
                targets_schema = _infer_schema(self._targets)
            return TensorDatasetSchema(features=features_schema, targets=targets_schema)
        except Exception as e:
            _logger.warning("Failed to infer schema for NumPy dataset. Exception: %s", e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        """
        return PyFuncInputsOutputs(self._features, self._targets)

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.sklearn.evalute().
        """
        return EvaluationDataset(
            data=self._features,
            targets=self._targets,
            path=path,
            feature_names=feature_names,
        )


def from_numpy(
    features: Union[np.ndarray, Dict[str, np.ndarray]],
    source: Union[str, DatasetSource] = None,
    targets: Union[np.ndarray, Dict[str, np.ndarray]] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> NumpyDataset:
    """
    Constructs a :py:class:`NumpyDataset <mlflow.data.numpy_dataset.NumpyDataset>` object from
    NumPy features, optional targets, and source. If the source is path like, then this will
    construct a DatasetSource object from the source path. Otherwise, the source is assumed to
    be a DatasetSource object.

    Args:
        features: NumPy features, represented as an np.ndarray or dictionary of named np.ndarrays.
        source: The source from which the numpy data was derived, e.g. a filesystem path, an S3 URI,
            an HTTPS URL, a delta table name with version, or spark table etc. ``source`` may be
            specified as a URI, a path-like string, or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`. If unspecified,
            the source is assumed to be the code location (e.g. notebook cell, script, etc.) where
            :py:func:`from_numpy <mlflow.data.from_numpy>` is being called.
        targets: Optional NumPy targets, represented as an np.ndarray or dictionary of named
            np.ndarrays.
        name: The name of the dataset. If unspecified, a name is generated.
        digest: The dataset digest (hash). If unspecified, a digest is computed automatically.

    .. code-block:: python
        :test:
        :caption: Basic Example

        import mlflow
        import numpy as np

        x = np.random.uniform(size=[2, 5, 4])
        y = np.random.randint(2, size=[2])
        dataset = mlflow.data.from_numpy(x, targets=y)

    .. code-block:: python
        :test:
        :caption: Dict Example

        import mlflow
        import numpy as np

        x = {
            "feature_1": np.random.uniform(size=[2, 5, 4]),
            "feature_2": np.random.uniform(size=[2, 5, 4]),
        }
        y = np.random.randint(2, size=[2])
        dataset = mlflow.data.from_numpy(x, targets=y)
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
    return NumpyDataset(
        features=features, source=resolved_source, targets=targets, name=name, digest=digest
    )
