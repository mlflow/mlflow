import hashlib
import json
import logging
from functools import cached_property
from typing import Any, Union, Optional, Mapping, Sequence, Dict

import datasets
import numpy as np
import pandas as pd

from mlflow.data.dataset import Dataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema

_logger = logging.getLogger(__name__)

_MAX_ROWS_FOR_DIGEST_COMPUTATION_AND_SCHEMA_INFERENCE = 10000


class HuggingFaceDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a HuggingFace dataset for use with MLflow Tracking.
    """

    def __init__(
        self,
        ds: datasets.Dataset,
        source: HuggingFaceDatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param ds: A Hugging Face dataset. Must be an instance of `datasets.Dataset`.
                   Other types, such as `datasets.DatasetDict`, are not supported.
        :param source: The source of the Hugging Face dataset.
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset.
        """
        if targets is not None and targets not in ds.column_names:
            raise MlflowException(
                f"The specified Hugging Face dataset does not contain the specified targets column"
                f" '{targets}'.",
                INVALID_PARAMETER_VALUE,
            )

        self._ds = ds
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """

        try:
            df = next(
                self._ds.to_pandas(
                    batch_size=_MAX_ROWS_FOR_DIGEST_COMPUTATION_AND_SCHEMA_INFERENCE, batched=True
                )
            )
        except Exception:
            if hasattr(self._ds, "_fingerprint") and self._ds._fingerprint is not None:
                return self._ds._fingerprint[:8]
            else:
                return hashlib.md5(np.int64(id(self._ds))).hexdigest()[:8]

        # drop object columns
        df = df.select_dtypes(exclude=["object"])
        # hash trimmed dataframe contents
        md5 = hashlib.md5(pd.util.hash_pandas_object(df).values)
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
    def ds(self) -> datasets.Dataset:
        """
        The Hugging Face `datasets.Dataset` instance.

        :return: The Hugging Face `datasets.Dataset` instance.
        """
        return self._ds

    @property
    def targets(self) -> Optional[str]:
        """
        The name of the Hugging Face dataset column containing targets.

        :return: The Hugging Face `datasets.Dataset` instance.
        """
        return self._targets

    @property
    def source(self) -> HuggingFaceDatasetSource:
        """
        Hugging Face dataset source information.

        :return: A HuggingFaceDatasetSource instance.
        """
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        """
        Summary statistics for the Hugging Face dataset, including the number of rows,
        size, and size in bytes.
        """
        return {
            "num_rows": self._ds.num_rows,
            "dataset_size": self._ds.dataset_size,
            "size_in_bytes": self._ds.size_in_bytes,
        }

    @cached_property
    def schema(self) -> Schema:
        """
        The MLflow ColSpec schema of the Hugging Face dataset.
        """
        try:
            df = next(
                self._ds.to_pandas(
                    batch_size=_MAX_ROWS_FOR_DIGEST_COMPUTATION_AND_SCHEMA_INFERENCE, batched=True
                )
            )
            return _infer_schema(df)
        except Exception as e:
            _logger._warning("Failed to infer schema for Hugging Face dataset. Exception: %s", e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        df = self._ds.to_pandas()
        if self._targets is not None:
            if self._targets not in df.columns:
                raise MlflowException(
                    f"Failed to convert Hugging Face dataset to pyfunc inputs and outputs because"
                    f" the pandas representation of the Hugging Face dataset does not contain the"
                    f" specified targets column '{self._targets}'.",
                    # This is an internal error because we should have validated the presence of
                    # the target column in the Hugging Face dataset at construction time
                    INTERNAL_ERROR,
                )
            inputs = df.drop(columns=self._targets)
            outputs = df[self._targets]
            return PyFuncInputsOutputs(inputs=inputs, outputs=outputs)
        else:
            return PyFuncInputsOutputs(inputs=df, outputs=None)


def from_huggingface(
    ds: datasets.Dataset,
    path: str,
    targets: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    task: Optional[Union[str, datasets.TaskTemplate]] = None,
    revision: Optional[Union[str, datasets.Version]] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> HuggingFaceDataset:
    if not isinstance(ds, datasets.Dataset):
        raise MlflowException(
            f"The specified Hugging Face dataset must be an instance of `datasets.Dataset`."
            f" Instead, found an instance of: {type(ds)}",
            INVALID_PARAMETER_VALUE,
        )

    source = HuggingFaceDatasetSource(
        path=path,
        config_name=ds.config_name,
        data_dir=data_dir,
        data_files=data_files,
        split=ds.split,
        revision=revision,
        task=task,
    )
    return HuggingFaceDataset(ds=ds, targets=targets, source=source, name=name, digest=digest)
