import json
import logging
from typing import List, Optional, Any, Dict, Union

import numpy as np
from functools import cached_property

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_numpy_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental
class NumpyDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a NumPy dataset for use with MLflow Tracking.
    """

    def __init__(
        self,
        features: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        source: DatasetSource,
        targets: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param features: A numpy array or list/dict of arrays containing dataset features.
        :param source: The source of the numpy dataset.
        :param targets: A numpy array or list/dict of arrays containing dataset targets. Optional
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
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
                "schema": json.dumps({"mlflow_tensorspec": self.schema.to_dict()})
                if self.schema
                else None,
                "profile": json.dumps(self.profile),
            }
        )
        return base_dict

    @property
    def source(self) -> DatasetSource:
        """
        The source of the dataset.
        """
        return self._source

    @property
    def features(self) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """
        The features of the dataset.
        """
        return self._features

    @property
    def targets(self) -> Optional[Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]]:
        """
        The targets of the dataset. May be None if no targets are available.
        """
        return self._targets

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        profile = {
            "features_shape": self._features.shape,
            "features_size": self._features.size,
            "features_nbytes": self._features.nbytes,
        }
        if self._targets is not None:
            profile.update(
                {
                    "targets_shape": self._targets.shape,
                    "targets_size": self._targets.size,
                    "targets_nbytes": self._targets.nbytes,
                }
            )

        return profile

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        An MLflow TensorSpec schema representing the tensor dataset
        """
        try:
            schema_dict = {
                "features": self._features,
            }
            if self._targets is not None:
                schema_dict["targets"] = self._targets
            return _infer_schema(schema_dict)
        except Exception as e:
            _logger.warning("Failed to infer schema for Numpy dataset. Exception: %s", e)
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


@experimental
def from_numpy(
    features: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    source: Union[str, DatasetSource],
    targets: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> NumpyDataset:
    """
    Constructs a NumpyDataset object from NumPy features, optional targets, and source.
    If the source is path like, then this will construct a DatasetSource object from the source
    path. Otherwise, the source is assumed to be a DatasetSource object.
    :param features: NumPy features, represented as an np.ndarray, list of np.ndarrays
                    or dictionary of named np.ndarrays.
    :param source: The source from which the NumPy data was derived, e.g. a filesystem
                    path, an S3 URI, an HTTPS URL etc. If source is not a path like string,
                    pass in a DatasetSource object directly.
    :param targets: Optional NumPy targets, represented as an np.ndarray, list of
                    np.ndarrays or dictionary of named np.ndarrays.
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: A dataset digest (hash). If unspecified, a digest is computed
                    automatically.
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source = resolve_dataset_source(source)
    return NumpyDataset(
        features=features, source=resolved_source, targets=targets, name=name, digest=digest
    )
