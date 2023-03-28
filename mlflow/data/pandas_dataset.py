import hashlib
import base64
import json
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema

from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource

MAX_ROWS = 10000


class PandasDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        source: FileSystemDatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._data = data
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        # drop object columns
        obj = self._data.select_dtypes(exclude=["object"])
        trimmed_df = obj.head(MAX_ROWS)
        # hash trimmed dataframe contents
        hash_md5 = hashlib.md5(pd.util.hash_pandas_object(trimmed_df).values)
        # hash dataframe dimensions
        n_rows = len(obj)
        hash_md5.update(np.int64(n_rows))
        # hash column names
        columns = obj.columns
        for x in columns:
            hash_md5.update(x.encode())
        return base64.b64encode(hash_md5.digest()).decode("ascii")

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
                "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()}),
                "profile": json.dumps(self.profile),
            }
        )
        return base_dict

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def source(self) -> FileSystemDatasetSource:
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        return {
            "length": len(self._data),
        }

    @property
    def schema(self) -> Schema:
        return _infer_schema(self._data)
