import json
import logging
from functools import cached_property
from typing import Any, Union, Optional, Mapping, Sequence, Dict, TYPE_CHECKING


from mlflow.data.dataset import Dataset
from mlflow.data.digest_utils import compute_pandas_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_MAX_ROWS_FOR_DIGEST_COMPUTATION_AND_SCHEMA_INFERENCE = 10000

if TYPE_CHECKING:
    import datasets


@experimental
class HuggingFaceDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a HuggingFace dataset for use with MLflow Tracking.
    """

    def __init__(
        self,
        ds: "datasets.Dataset",
        source: HuggingFaceDatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param ds: A Hugging Face dataset. Must be an instance of `datasets.Dataset`.
                   Other types, such as :py:class:`datasets.DatasetDict`, are not supported.
        :param source: The source of the Hugging Face dataset.
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                       is automatically computed.
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
        df = next(
            self._ds.to_pandas(
                batch_size=_MAX_ROWS_FOR_DIGEST_COMPUTATION_AND_SCHEMA_INFERENCE, batched=True
            )
        )
        return compute_pandas_digest(df)

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """
        return {
            **base_dict,
            "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()})
            if self.schema
            else None,
            "profile": json.dumps(self.profile),
        }

    @property
    def ds(self) -> "datasets.Dataset":
        """
        The Hugging Face ``datasets.Dataset`` instance.

        :return: The Hugging Face ``datasets.Dataset`` instance.
        """
        return self._ds

    @property
    def targets(self) -> Optional[str]:
        """
        The name of the Hugging Face dataset column containing targets (labels) for supervised
        learning.

        :return: The string name of the Hugging Face dataset column containing targets.
        """
        return self._targets

    @property
    def source(self) -> HuggingFaceDatasetSource:
        """
        Hugging Face dataset source information.

        :return: A :py:class:`mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource`
                 instance.
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
    def schema(self) -> Optional[Schema]:
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
            _logger.warning("Failed to infer schema for Hugging Face dataset. Exception: %s", e)
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

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return EvaluationDataset(
            data=self._ds.to_pandas(),
            targets=self._targets,
            path=path,
            feature_names=feature_names,
        )


@experimental
def from_huggingface(
    ds,
    path: str = None,
    targets: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    revision=None,
    task=None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> HuggingFaceDataset:
    """
    Given a Hugging Face ``datasets.Dataset``, constructs an MLflow :py:class:`HuggingFaceDataset`
    object for use with MLflow Tracking.

    :param ds: A Hugging Face dataset. Must be an instance of ``datasets.Dataset``.
               Other types, such as ``datasets.DatasetDict``, are not supported.
    :param path: The path of the Hugging Face dataset used to construct the source. This is used by
                 the ``datasets.load_dataset()`` function to reload the dataset upon request via
                 :py:func:`HuggingFaceDataset.source.load()
                 <mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource.load>`.
                 If no path is specified, a CodeDatasetSource is used, which will source
                 information from the run context.
    :param targets: The name of the Hugging Face ``dataset.Dataset`` column containing targets
                    (labels) for supervised learning.
    :param data_dir: The `data_dir` of the Hugging Face dataset configuration. This is used by the
                     ``datasets.load_dataset()`` function to reload the dataset upon request via
                     :py:func:`HuggingFaceDataset.source.load()
                     <mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource.load>`.
    :param data_files: Paths to source data file(s) for the Hugging Face dataset configuration.
                       This is used by the ``datasets.load_dataset()`` function to reload the
                       dataset upon request via :py:func:`HuggingFaceDataset.source.load()
                       <mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource.load>`.
    :param revision: Version of the dataset script to load. This is used by the
                     ``datasets.load_dataset()`` function to reload the dataset upon request via
                     :py:func:`HuggingFaceDataset.source.load()
                     <mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource.load>`.
    :param task: The task to prepare the Hugging Face dataset for during training and evaluation.
                 This is used by the ``datasets.load_dataset()`` function to reload the dataset
                 upon request via :py:func:`HuggingFaceDataset.source.load()
                 <mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource.load>`.
    :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                 automatically generated.
    :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                   is automatically computed.
    """
    import datasets
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.tracking.context import registry

    if not isinstance(ds, datasets.Dataset):
        raise MlflowException(
            f"The specified Hugging Face dataset must be an instance of ``datasets.Dataset``."
            f" Instead, found an instance of: {type(ds)}",
            INVALID_PARAMETER_VALUE,
        )

    # if arguments for source are passed in directly, use them.
    # otherwise construct a CodeDatasetSource.
    if path is not None:
        source = HuggingFaceDatasetSource(
            path=path,
            config_name=ds.config_name,
            data_dir=data_dir,
            data_files=data_files,
            split=ds.split,
            revision=revision,
            task=task,
        )
    else:
        context_tags = registry.resolve_tags()
        source = CodeDatasetSource(tags=context_tags)
    return HuggingFaceDataset(ds=ds, targets=targets, source=source, name=name, digest=digest)
