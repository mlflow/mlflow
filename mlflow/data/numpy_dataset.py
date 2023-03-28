import hashlib
import json
from typing import List, Optional, Any, Dict, Union

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class NumpyDataset(Dataset):
    def __init__(
        self,
        features: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        source: FileSystemDatasetSource,
        targets: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        TODO: Numpy docs
        """
        self._features = features
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        MAX_ROWS = 10000

        flattened_data = self._features.flatten()
        trimmed_data = flattened_data[0:MAX_ROWS]

        # hash trimmed array contents
        try:
            md5 = hashlib.md5(pd.util.hash_array(trimmed_data))
        except TypeError:
            md5 = hashlib.md5(np.int64(trimmed_data.size))
        # hash full array dimensions
        for x in self._features.shape:
            md5.update(np.int64(x))
        # TODO: Make this a normalize_hash function (truncation)
        return md5.hexdigest()[:8]

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
    def source(self) -> FileSystemDatasetSource:
        """
        TODO: Numpy docs
        """
        return self._source

    @property
    def features(self) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        return self._features

    @property
    def targets(self) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        return self._targets

    @property
    def profile(self) -> Optional[Any]:
        """
        TODO: Numpy docs
        """
        return {
            "shape": self._features.shape,
        }

    @property
    def schema(self) -> Schema:
        """
        An MLflow TensorSpec schema representing the tensor dataset
        """
        # TODO: Error handling
        return _infer_schema(self._features)


def from_numpy(
    features: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    source: str,
    targets: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> NumpyDataset:
    """
    :param features: NumPy features, represented as an np.ndarray, list of np.ndarrays
                    or dictionary of named np.ndarrays.
    :param source: The source from which the NumPy data was derived, e.g. a filesystem
                    path, an S3 URI, an HTTPS URL etc.
    :param targets: Optional NumPy targets, represented as an np.ndarray, list of
                    np.ndarrays or dictionary of named np.ndarrays.
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: A dataset digest (hash). If unspecified, a digest is computed
                    automatically.
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source: FileSystemDatasetSource = resolve_dataset_source(
        source, candidate_sources=[FileSystemDatasetSource]
    )
    return NumpyDataset(
        features=features, source=resolved_source, targets=targets, name=name, digest=digest
    )
