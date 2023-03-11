import hashlib
import json
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.types.utils import _infer_schema
from mlflow.types import Schema 


class PandasDataset(Dataset):

    def __init__(
        self, df: pd.DataFrame, source: FileSystemDatasetSource, name: Optional[str] = None, digest: Optional[str] = None
    ):
        """
        TODO: Pandas docs
        """
        self._df = df
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        MAX_ROWS = 10000

        # drop object columns
        df = self._df.select_dtypes(exclude=['object'])
        trimmed_df = df.head(MAX_ROWS)
        # hash trimmed dataframe contents
        md5 = hashlib.md5(pd.util.hash_pandas_object(trimmed_df).values)
        # hash dataframe dimensions
        n_rows = len(df)
        md5.update(np.int64(n_rows))
        # hash column names
        columns = df.columns
        for x in columns:
          md5.update(x.encode())
        # TODO: Make this a normalize_hash function (truncation)
        return md5.hexdigest()[:8]

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
            "schema": self.schema.to_json(),
            "size": json.dumps(self.size),
        })
        return base_dict

    @property
    def digest(self) -> str:
        """
        The digest (hash) of the dataset, e.g. "498c7496"
        """
        return self._digest

    @property
    def source(self) -> FileSystemDatasetSource:
        """
        TODO: Pandas docs
        """
        return self._source

    @property
    def size(self) -> Optional[Any]:
        """
        TODO: Pandas docs
        """
        return {
            "num_rows": len(self._df),
            "num_elements": int(self._df.size),
        }

    @property
    def schema(self) -> Schema:
        """
        TODO: Pandas docs
        """
        # TODO: Error handling
        return _infer_schema(self._df)


def from_pandas(df: pd.DataFrame, source: str, name: Optional[str] = None, digest: Optional[str] = None) -> PandasDataset:
    """
    TODO: Pandas docs
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source: FileSystemDatasetSource = resolve_dataset_source(source, candidate_sources=[FileSystemDatasetSource])
    return PandasDataset(df=df, source=resolved_source, name=name, digest=digest)
