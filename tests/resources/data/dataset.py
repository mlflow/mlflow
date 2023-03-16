import hashlib
import base64
import json
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema

from tests.resources.data.dataset_source import TestDatasetSource


class TestDataset(Dataset):

    def __init__(
        self, data_list: List[int], source: TestDatasetSource, name: Optional[str] = None, digest: Optional[str] = None
    ):
        self._data_list = data_list
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        hash_md5 = hashlib.md5()
        for hash_part in pd.util.hash_array(np.array(self._data_list)):
            hash_md5.update(hash_part)
        return base64.b64encode(hash_md5.digest()).decode("ascii")

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), size
                 (optional).
        """
        base_dict.update({
            "schema": json.dumps({
                "mlflow_colspec": self.schema.to_dict()
            }),
            "size": json.dumps(self.size),
        })
        return base_dict

    @property
    def data_list(self) -> List[int]:
        return self._data_list

    @property
    def source(self) -> TestDatasetSource:
        return self._source

    @property
    def size(self) -> Optional[Any]:
        return {
            "length": len(self._data_list),
        }

    @property
    def schema(self) -> Schema:
        return _infer_schema(np.array(self._data_list))
