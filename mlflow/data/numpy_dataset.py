import hashlib
import json
from typing import Optional, Any, Dict

import numpy as np

from mlflow.data.dataset import Dataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class NumpyDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        source: FileSystemDatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        TODO: Numpy docs
        """
        self._data = data
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        MAX_ROWS = 10000

        flattened_data = self._data.flatten()
        trimmed_data = flattened_data[0:MAX_ROWS]

        # hash trimmed array contents
        try:
            md5 = hashlib.md5(pd.util.hash_array(trimmed_data))
        except TypeError:
            md5 = hashlib.md5(np.int64(obj.size))
        # hash full array dimensions
        for x in self._data.shape:
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
    def profile(self) -> Optional[Any]:
        """
        TODO: Numpy docs
        """
        return {
            "shape": self._data.shape,
        }

    @property
    def schema(self) -> Schema:
        """
        TODO: Numpy docs
        """
        # TODO: Error handling
        return _infer_schema(self._data)


def from_numpy(
    data: np.ndarray, source: str, name: Optional[str] = None, digest: Optional[str] = None
) -> NumpyDataset:
    """
    TODO: Numpy docs
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source: FileSystemDatasetSource = resolve_dataset_source(
        source, candidate_sources=[FileSystemDatasetSource]
    )
    return NumpyDataset(data=data, source=resolved_source, name=name, digest=digest)
