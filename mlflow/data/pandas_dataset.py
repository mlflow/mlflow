import hashlib
import json
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class PandasDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        source: FileSystemDatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
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
        df = self._df.select_dtypes(exclude=["object"])
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
    def source(self) -> FileSystemDatasetSource:
        """
        TODO: Pandas docs
        """
        return self._source

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def profile(self) -> Optional[Any]:
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
        An MLflow ColSpec schema representing the columnar dataset
        """
        return _infer_schema(self._df)


def from_pandas(
    df: pd.DataFrame, source: str, name: Optional[str] = None, digest: Optional[str] = None
) -> PandasDataset:
    """
    :param df: A Pandas DataFrame
    :param source: The source from which the DataFrame was derived. E.g. a Spark table
                    name, a delta table name with version, a distributed filesystem path or
                    URI, etc.
    :param targets: An optional target column name or list of target column names for
                    supervised training. The columns must be present in the dataframe
                    (`df`).
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: A dataset digest (hash). If unspecified, a digest is computed
                    automatically.
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source: FileSystemDatasetSource = resolve_dataset_source(
        source, candidate_sources=[FileSystemDatasetSource]
    )
    return PandasDataset(df=df, source=resolved_source, name=name, digest=digest)
