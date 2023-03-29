import json
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
from pyspark.sql import DataFrame

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class SparkDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        source: DatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._df = df
        super().__init__(source=source, name=name, digest=digest)

    @staticmethod
    def _parse_logical_plan(df):
        d = json.loads(df._jdf.queryExecution().logical().toJSON())

        def purge_key(input_dict, key):
            if isinstance(input_dict, dict):
                return {k: purge_key(v, key) for k, v in input_dict.items() if k != key}

            elif isinstance(input_dict, list):
                return [purge_key(element, key) for element in input_dict]

            else:
                return input_dict

        d = purge_key(d, "exprId")
        d = purge_key(d, "resultId")

        return d

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        parsed_plan = SparkDataset.parse_logical_plan(self._df)
        plan_str = json.dumps(parsed_plan)
        return hashlib.md5(plan_str.encode("utf-8")).hexdigest()[:8]

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
    def df(self) -> DataFrame:
        return self._df

    @property
    def targets(self) -> Optional[str]:
        return self._targets

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        # TODO: Include an approximate count of records from the table source. We don't
        # want to compute the full count, which could be quite slow
        return {}

    @property
    def schema(self) -> Schema:
        return _infer_schema(self._df)


def load_delta(
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    table_version: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> SparkDataset:
    source = DeltaDatasetSource(
        path=path,
        delta_table_name=table_name,
        delta_table_version=table_version
    )
    df: DataFrame = source.load()
    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )


def from_spark(
    df: DataFrame, source: Any = None,
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    table_version: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
):
   # Verify that either path or table_name with optional table version are specified, but not both

    if path is not None:
        source = resolve_dataset_source(path, candidate_sources=[DeltaDatasetSource, SparkDatasetSource, FileSystemDatasetSource])
    elif table_name is not None:
        if table_version is not None:
            source = DeltaDatasetSource(
                delta_table_name=table_name,
                delta_table_version=table_version,
            )
        else:
            source = resolve_dataset_source(path, candidate_sources=[DeltaDatasetSource, SparkDatasetSource])

    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )


