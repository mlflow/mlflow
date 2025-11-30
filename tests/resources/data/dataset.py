import base64
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema

from tests.resources.data.dataset_source import SampleDatasetSource


class SampleDataset(Dataset):
    def __init__(
        self,
        data_list: list[int],
        source: SampleDatasetSource,
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
        hash_md5 = hashlib.md5(usedforsecurity=False)
        for hash_part in pd.util.hash_array(np.array(self._data_list)):
            hash_md5.update(hash_part)
        return base64.b64encode(hash_md5.digest()).decode("ascii")

    def to_dict(self) -> dict[str, str]:
        """
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
    def data_list(self) -> list[int]:
        return self._data_list

    @property
    def source(self) -> SampleDatasetSource:
        return self._source

    @property
    def profile(self) -> Any | None:
        return {
            "length": len(self._data_list),
        }

    @property
    def schema(self) -> Schema:
        return _infer_schema(np.array(self._data_list))
