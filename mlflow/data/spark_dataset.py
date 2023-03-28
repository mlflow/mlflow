import json
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
from pyspark.sql import DataFrame

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class SparkDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        source: DatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._df = df
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        semantic_hash = self._df.semanticHash()
        print("SEM HASH", semantic_hash)
        md5_hash = hashlib.md5()
        md5_hash.update(np.int64(semantic_hash))
        for column in self._df.columns:
            md5_hash.update(column.encode())
        return md5_hash.hexdigest()[:8]

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
    def data_list(self) -> List[int]:
        return self._data_list

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        return {
            "length": len(self._data_list),
        }

    @property
    def schema(self) -> Schema:
        return _infer_schema(self._data_list)
