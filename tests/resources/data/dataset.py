import base64
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils import insecure_hash

from tests.resources.data.dataset_source import SampleDatasetSource


class SampleDataset(Dataset):
    def __init__(
        self,
        data_list: List[int],
        source: SampleDatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._data_list = data_list
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        hash_md5 = insecure_hash.md5()
        for hash_part in pd.util.hash_array(np.array(self._data_list)):
            hash_md5.update(hash_part)
        return base64.b64encode(hash_md5.digest()).decode("ascii")

    def to_dict(self) -> Dict[str, str]:
        """
        Args:
            base_dict: A string dictionary of base information about the
                dataset, including: name, digest, source, and source
                type.

        Returns:
            A string dictionary containing the following fields: name,
            digest, source, source type, schema (optional), profile
            (optional).
        """
        config = super().to_dict()
        config.update(
            {
                "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()}),
                "profile": json.dumps(self.profile),
            }
        )
        return config

    @property
    def data_list(self) -> List[int]:
        return self._data_list

    @property
    def source(self) -> SampleDatasetSource:
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        return {
            "length": len(self._data_list),
        }

    @property
    def schema(self) -> Schema:
        return _infer_schema(np.array(self._data_list))
