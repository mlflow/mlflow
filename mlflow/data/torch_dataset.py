import json
import logging
from functools import cached_property
from typing import Any

import numpy as np

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import MAX_ROWS, get_normalized_md5_digest
from mlflow.data.schema import TensorDatasetSchema
from mlflow.types.utils import _infer_schema

_logger = logging.getLogger(__name__)

# Number of dataset items sampled when computing the digest and schema.
_SAMPLE_SIZE = 10


def _tensor_to_numpy(tensor):
    """Convert a torch.Tensor to a numpy array without importing torch at module level."""
    return tensor.detach().cpu().numpy()


def _sample_dataset(dataset, n: int):
    """
    Return up to *n* items from a ``torch.utils.data.Dataset``.

    Returns a list of raw items (each item may be a Tensor, tuple, dict, etc.).
    """
    items = []
    try:
        size = len(dataset)
        indices = range(min(n, size))
    except TypeError:
        # IterableDataset — iterate until we have enough
        indices = None

    if indices is not None:
        for i in indices:
            try:
                items.append(dataset[i])
            except Exception:
                break
    else:
        for i, item in enumerate(dataset):
            if i >= n:
                break
            items.append(item)

    return items


def _item_to_numpy(item):
    """
    Recursively convert a dataset item to numpy.

    Handles Tensor, tuple/list of Tensors, and dict of Tensors.
    Returns None if conversion is not possible.
    """
    import torch  # lazy import — torch is required but not a top-level dep

    if isinstance(item, torch.Tensor):
        return _tensor_to_numpy(item)
    elif isinstance(item, (tuple, list)):
        arrays = []
        for v in item:
            if isinstance(v, torch.Tensor):
                arrays.append(_tensor_to_numpy(v))
            elif isinstance(v, np.ndarray):
                arrays.append(v)
        return arrays if arrays else None
    elif isinstance(item, dict):
        result = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                result[str(k)] = _tensor_to_numpy(v)
            elif isinstance(v, np.ndarray):
                result[str(k)] = v
        return result if result else None
    elif isinstance(item, np.ndarray):
        return item
    return None


class TorchDataset(Dataset):
    """
    Represents a ``torch.utils.data.Dataset`` for use with MLflow Tracking.

    Use :py:func:`mlflow.data.from_torch` to construct a ``TorchDataset`` from a PyTorch
    dataset or data loader, then pass the result to :py:func:`mlflow.log_input` to record
    it in the active run.
    """

    def __init__(
        self,
        dataset,
        source: DatasetSource,
        targets=None,
        name: str | None = None,
        digest: str | None = None,
        _dataloader_metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            dataset: A ``torch.utils.data.Dataset`` instance containing the features.
            source: The source of the dataset.
            targets: An optional ``torch.utils.data.Dataset`` instance containing the targets.
            name: The name of the dataset. If unspecified, a name is generated.
            digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                is computed automatically.
            _dataloader_metadata: Optional dict of DataLoader-level metadata (e.g.
                ``batch_size``, ``num_workers``) captured by :py:func:`from_torch` when
                a ``DataLoader`` is passed. Not intended for direct use.
        """
        self._dataset = dataset
        self._targets = targets
        self._dataloader_metadata = _dataloader_metadata or {}
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """Computes a digest by sampling up to ``_SAMPLE_SIZE`` items from the dataset."""
        import pandas as pd  # lazy import consistent with other MLflow dataset digest helpers

        hashable_elements: list = []

        def _hash_item(item):
            arr = _item_to_numpy(item)
            if arr is None:
                return
            if isinstance(arr, list):
                arrays = arr
            elif isinstance(arr, dict):
                arrays = list(arr.values())
            else:
                arrays = [arr]
            for a in arrays:
                flat = a.flatten()[: MAX_ROWS]
                try:
                    hashable_elements.append(pd.util.hash_array(flat))
                except TypeError:
                    hashable_elements.append(np.int64(flat.size))
                hashable_elements.extend(np.int64(x) for x in a.shape)

        for item in _sample_dataset(self._dataset, _SAMPLE_SIZE):
            _hash_item(item)

        # Include the dataset class name so that two identically-valued datasets of different
        # types produce different digests.
        hashable_elements.append(type(self._dataset).__name__.encode())

        if not hashable_elements:
            # Fall back to hashing just the class name if we can't sample any data.
            hashable_elements.append(type(self._dataset).__name__.encode())

        return get_normalized_md5_digest(hashable_elements)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable config dictionary for the dataset."""
        schema = json.dumps(self.schema.to_dict()) if self.schema else None
        profile = self.profile
        config = super().to_dict()
        config.update({
            "schema": schema,
            "profile": json.dumps(profile) if profile is not None else None,
        })
        return config

    @property
    def dataset(self):
        """The underlying ``torch.utils.data.Dataset``."""
        return self._dataset

    @property
    def source(self) -> DatasetSource:
        """The source of the dataset."""
        return self._source

    @property
    def targets(self):
        """Optional targets dataset. May be ``None``."""
        return self._targets

    @property
    def profile(self) -> dict[str, Any] | None:
        """
        A profile of the dataset containing summary information such as size and type.
        """
        prof: dict[str, Any] = {
            "dataset_type": type(self._dataset).__name__,
        }
        try:
            prof["dataset_size"] = len(self._dataset)
        except TypeError:
            # IterableDataset — size unknown
            pass
        if self._targets is not None:
            prof["targets_type"] = type(self._targets).__name__
            try:
                prof["targets_size"] = len(self._targets)
            except TypeError:
                pass
        if self._dataloader_metadata:
            prof.update(self._dataloader_metadata)
        return prof

    @cached_property
    def schema(self) -> TensorDatasetSchema | None:
        """
        A :py:class:`TensorDatasetSchema <mlflow.data.schema.TensorDatasetSchema>` inferred
        from the first item of the dataset.  Returns ``None`` if the schema cannot be inferred.
        """
        try:
            items = _sample_dataset(self._dataset, 1)
            if not items:
                return None
            arr = _item_to_numpy(items[0])
            if arr is None:
                return None

            if isinstance(arr, np.ndarray):
                features_schema = _infer_schema(arr)
            elif isinstance(arr, list):
                features_schema = _infer_schema(
                    {str(i): a for i, a in enumerate(arr)}
                )
            elif isinstance(arr, dict):
                features_schema = _infer_schema(arr)
            else:
                return None

            targets_schema = None
            if self._targets is not None:
                t_items = _sample_dataset(self._targets, 1)
                if t_items:
                    t_arr = _item_to_numpy(t_items[0])
                    if t_arr is not None:
                        if isinstance(t_arr, np.ndarray):
                            targets_schema = _infer_schema(t_arr)
                        elif isinstance(t_arr, list):
                            targets_schema = _infer_schema(
                                {str(i): a for i, a in enumerate(t_arr)}
                            )
                        elif isinstance(t_arr, dict):
                            targets_schema = _infer_schema(t_arr)

            return TensorDatasetSchema(features=features_schema, targets=targets_schema)
        except Exception as e:
            _logger.warning("Failed to infer schema for TorchDataset. Exception: %s", e)
            return None


def from_torch(
    dataset,
    source: str | DatasetSource | None = None,
    targets=None,
    name: str | None = None,
    digest: str | None = None,
) -> TorchDataset:
    """
    Constructs a :py:class:`TorchDataset <mlflow.data.torch_dataset.TorchDataset>` from a
    PyTorch dataset or data loader.

    If a ``DataLoader`` is passed, the underlying dataset is extracted and DataLoader
    metadata (``batch_size``, ``num_workers``) is preserved in the dataset profile.

    Args:
        dataset: A ``torch.utils.data.Dataset`` or ``torch.utils.data.DataLoader``.
        source: The source from which the data was derived, e.g. a filesystem path, S3 URI,
            HTTPS URL, etc.  May be a URI string or a
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>` instance.
            If ``None``, the source is inferred from the run context (code location).
        targets: An optional ``torch.utils.data.Dataset`` containing the targets.
        name: The name of the dataset. If unspecified, a name is generated.
        digest: The dataset digest (hash). If unspecified, a digest is computed automatically.

    Returns:
        A :py:class:`TorchDataset <mlflow.data.torch_dataset.TorchDataset>`.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        import torch
        from torch.utils.data import TensorDataset

        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,))
        ds = TensorDataset(X, y)

        dataset = mlflow.data.from_torch(ds, targets=None)
        with mlflow.start_run():
            mlflow.log_input(dataset, context="training")
    """
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required to use `mlflow.data.from_torch`. "
            "Install it with `pip install torch`."
        ) from e

    # Accept DataLoader — extract the underlying dataset and capture loader metadata.
    dataloader_profile: dict[str, Any] = {}
    if isinstance(dataset, torch.utils.data.DataLoader):
        loader = dataset
        dataset = loader.dataset
        if loader.batch_size is not None:
            dataloader_profile["batch_size"] = loader.batch_size
        dataloader_profile["num_workers"] = loader.num_workers

    if not isinstance(dataset, torch.utils.data.Dataset):
        raise TypeError(
            f"'dataset' must be a torch.utils.data.Dataset or torch.utils.data.DataLoader. "
            f"Got: {type(dataset)}."
        )

    if source is not None:
        if isinstance(source, DatasetSource):
            resolved_source = source
        else:
            resolved_source = resolve_dataset_source(source)
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)

    return TorchDataset(
        dataset=dataset,
        source=resolved_source,
        targets=targets,
        name=name,
        digest=digest,
        _dataloader_metadata=dataloader_profile if dataloader_profile else None,
    )
