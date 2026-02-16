import json
from typing import Any

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow_test_plugin.dummy_dataset_source import DummyDatasetSource


class DummyDataset(Dataset):
    def __init__(
        self,
        data_list: list[int],
        source: DummyDatasetSource,
        name: str | None = None,
        digest: str | None = None,
    ):
        self._data_list = data_list
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        return pd.util.hash_array(np.ndarray(self._data_list))

    def _to_dict(self, base_dict: dict[str, str]) -> dict[str, str]:
        """
        Args:
            base_dict: A string dictionary of base information about the
                dataset, including: name, digest, source, and source type.

        Returns:
            A string dictionary containing the following fields: name,
            digest, source, source type, schema (optional), profile
            (optional).
        """
        return {
            **base_dict,
            "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()}),
            "profile": json.dumps(self.profile),
        }

    @property
    def data_list(self) -> list[int]:
        return self._data_list

    @property
    def source(self) -> DummyDatasetSource:
        return self._source

    @property
    def profile(self) -> Any | None:
        return {
            "length": len(self._data_list),
        }

    @property
    def schema(self) -> Schema:
        return _infer_schema(self._data_list)


def from_dummy(
    data_list: list[int], source: str, name: str | None = None, digest: str | None = None
) -> DummyDataset:
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source: DummyDatasetSource = resolve_dataset_source(
        source, candidate_sources=[DummyDatasetSource]
    )
    return DummyDataset(data_list=data_list, source=resolved_source, name=name, digest=digest)
