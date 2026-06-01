"""
Tests for mlflow.data.torch_dataset (TorchDataset + from_torch).

Regression tests for https://github.com/mlflow/mlflow/issues/11764.
"""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.data
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.schema import TensorDatasetSchema
from mlflow.data.torch_dataset import TorchDataset, from_torch

from tests.resources.data.dataset_source import SampleDatasetSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor_dataset(n: int = 20, features: int = 4, classes: int = 3):
    """Return a TensorDataset with random features and integer labels."""
    X = torch.randn(n, features)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(X, y)


def _make_source():
    return SampleDatasetSource._resolve("test:/my/uri")


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------


def test_basic_construction():
    ds = _make_tensor_dataset()
    source = _make_source()
    mlflow_dataset = TorchDataset(dataset=ds, source=source, name="my_dataset")

    assert mlflow_dataset.name == "my_dataset"
    assert mlflow_dataset.digest is not None
    assert len(mlflow_dataset.digest) == 8
    assert mlflow_dataset.dataset is ds
    assert mlflow_dataset.targets is None


def test_digest_is_deterministic():
    ds = _make_tensor_dataset(n=10)
    source = _make_source()
    d1 = TorchDataset(dataset=ds, source=source)
    d2 = TorchDataset(dataset=ds, source=source)
    assert d1.digest == d2.digest


def test_digest_differs_for_different_data():
    torch.manual_seed(0)
    ds1 = _make_tensor_dataset(n=10)
    torch.manual_seed(99)
    ds2 = _make_tensor_dataset(n=10)
    source = _make_source()
    assert TorchDataset(dataset=ds1, source=source).digest != TorchDataset(
        dataset=ds2, source=source
    ).digest


def test_explicit_digest_is_preserved():
    ds = _make_tensor_dataset()
    source = _make_source()
    mlflow_dataset = TorchDataset(dataset=ds, source=source, digest="abcd1234")
    assert mlflow_dataset.digest == "abcd1234"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def test_profile_contains_type_and_size():
    ds = _make_tensor_dataset(n=15)
    source = _make_source()
    profile = TorchDataset(dataset=ds, source=source).profile

    assert profile["dataset_type"] == "TensorDataset"
    assert profile["dataset_size"] == 15


def test_profile_with_targets_dataset():
    X = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))
    feat_ds = TensorDataset(X)
    tgt_ds = TensorDataset(y)
    source = _make_source()
    profile = TorchDataset(dataset=feat_ds, source=source, targets=tgt_ds).profile

    assert "targets_type" in profile
    assert profile["targets_size"] == 10


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_inferred_from_tensor_dataset():
    ds = _make_tensor_dataset(n=5, features=4)
    source = _make_source()
    mlflow_dataset = TorchDataset(dataset=ds, source=source)

    schema = mlflow_dataset.schema
    # TensorDataset returns a tuple (X, y) — schema should be inferred
    assert schema is not None
    assert isinstance(schema, TensorDatasetSchema)


def test_schema_is_none_when_not_inferable():
    class NonTensorDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, idx):
            return "some_string_item"

    ds = NonTensorDataset()
    source = _make_source()
    # Should warn but not raise; schema returns None
    mlflow_dataset = TorchDataset(dataset=ds, source=source)
    assert mlflow_dataset.schema is None


# ---------------------------------------------------------------------------
# to_dict / to_json round-trip
# ---------------------------------------------------------------------------


def test_to_dict_keys():
    ds = _make_tensor_dataset()
    source = _make_source()
    mlflow_dataset = TorchDataset(dataset=ds, source=source, name="train")

    d = mlflow_dataset.to_dict()
    assert set(d.keys()) <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert d["name"] == "train"
    assert d["digest"] == mlflow_dataset.digest
    assert d["source"] == mlflow_dataset.source.to_json()
    assert d["source_type"] == mlflow_dataset.source._get_source_type()


def test_to_json_is_valid_json():
    ds = _make_tensor_dataset()
    source = _make_source()
    mlflow_dataset = TorchDataset(dataset=ds, source=source)
    parsed = json.loads(mlflow_dataset.to_json())
    assert "name" in parsed
    assert "digest" in parsed


def test_profile_serialized_in_to_dict():
    ds = _make_tensor_dataset(n=8)
    source = _make_source()
    mlflow_dataset = TorchDataset(dataset=ds, source=source)

    d = mlflow_dataset.to_dict()
    profile_parsed = json.loads(d["profile"])
    assert profile_parsed["dataset_size"] == 8
    assert profile_parsed["dataset_type"] == "TensorDataset"


# ---------------------------------------------------------------------------
# from_torch() — Dataset path
# ---------------------------------------------------------------------------


def test_from_torch_with_dataset():
    ds = _make_tensor_dataset(n=12)
    mlflow_dataset = from_torch(ds, source="test:/path/to/data")

    assert isinstance(mlflow_dataset, TorchDataset)
    assert mlflow_dataset.dataset is ds
    assert mlflow_dataset.targets is None
    assert mlflow_dataset.digest is not None


def test_from_torch_default_source_is_code():
    ds = _make_tensor_dataset()
    mlflow_dataset = from_torch(ds)
    assert isinstance(mlflow_dataset.source, CodeDatasetSource)


def test_from_torch_with_explicit_source_object():
    ds = _make_tensor_dataset()
    source = _make_source()
    mlflow_dataset = from_torch(ds, source=source)
    assert mlflow_dataset.source is source


# ---------------------------------------------------------------------------
# from_torch() — DataLoader path
# ---------------------------------------------------------------------------


def test_from_torch_with_dataloader_captures_batch_size():
    ds = _make_tensor_dataset(n=20)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    mlflow_dataset = from_torch(loader)

    assert isinstance(mlflow_dataset, TorchDataset)
    assert mlflow_dataset.dataset is ds
    profile = mlflow_dataset.profile
    assert profile["batch_size"] == 4
    assert "num_workers" in profile


def test_from_torch_with_dataloader_extracts_underlying_dataset():
    ds = _make_tensor_dataset(n=10)
    loader = DataLoader(ds, batch_size=2)
    mlflow_dataset = from_torch(loader)
    assert mlflow_dataset.dataset is ds


# ---------------------------------------------------------------------------
# from_torch() — invalid input
# ---------------------------------------------------------------------------


def test_from_torch_raises_on_invalid_type():
    with pytest.raises(TypeError, match="torch.utils.data.Dataset or torch.utils.data.DataLoader"):
        from_torch([1, 2, 3])


# ---------------------------------------------------------------------------
# mlflow.data module registration
# ---------------------------------------------------------------------------


def test_from_torch_registered_in_mlflow_data():
    # from_torch should be available directly on the mlflow.data module
    assert hasattr(mlflow.data, "from_torch")
    assert callable(mlflow.data.from_torch)


def test_from_torch_via_mlflow_data():
    ds = _make_tensor_dataset(n=5)
    mlflow_dataset = mlflow.data.from_torch(ds)
    assert isinstance(mlflow_dataset, TorchDataset)


# ---------------------------------------------------------------------------
# log_input integration
# ---------------------------------------------------------------------------


def test_log_input_with_torch_dataset(tmp_path):
    ds = _make_tensor_dataset(n=10)
    mlflow_dataset = from_torch(ds)

    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    with mlflow.start_run() as run:
        mlflow.log_input(mlflow_dataset, context="training")

    client = mlflow.tracking.MlflowClient()
    inputs = client.get_run(run.info.run_id).inputs
    assert len(inputs.dataset_inputs) == 1
    logged = inputs.dataset_inputs[0].dataset
    assert logged.name == mlflow_dataset.name
    assert logged.digest == mlflow_dataset.digest
