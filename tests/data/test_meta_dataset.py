import json
from unittest.mock import patch

import pytest

from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.uc_volume_dataset_source import UCVolumeDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema


@pytest.mark.parametrize(
    ("dataset_source_class", "path"),
    [
        (HTTPDatasetSource, "test:/my/test/uri"),
        (DeltaDatasetSource, "fake/path/to/delta"),
        (HuggingFaceDatasetSource, "databricks/databricks-dolly-15k"),
    ],
)
def test_create_meta_dataset_from_source(dataset_source_class, path):
    source = dataset_source_class(path)
    dataset = MetaDataset(source=source)

    json_str = dataset.to_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["digest"] is not None
    assert path in parsed_json["source"]
    assert parsed_json["source_type"] == dataset_source_class._get_source_type()


@pytest.mark.parametrize(
    ("dataset_source_class", "path"),
    [
        (HTTPDatasetSource, "test:/my/test/uri"),
        (DeltaDatasetSource, "fake/path/to/delta"),
        (HuggingFaceDatasetSource, "databricks/databricks-dolly-15k"),
    ],
)
def test_create_meta_dataset_from_source_with_schema(dataset_source_class, path):
    source = dataset_source_class(path)
    schema = Schema(
        [
            ColSpec(type=DataType.long, name="foo"),
            ColSpec(type=DataType.integer, name="bar"),
        ]
    )
    dataset = MetaDataset(source=source, schema=schema)

    json_str = dataset.to_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["digest"] is not None
    assert path in parsed_json["source"]
    assert parsed_json["source_type"] == dataset_source_class._get_source_type()
    assert json.loads(parsed_json["schema"])["mlflow_colspec"] == schema.to_dict()


def test_meta_dataset_digest():
    http_source = HTTPDatasetSource("test:/my/test/uri")
    dataset1 = MetaDataset(source=http_source)
    schema = Schema(
        [
            ColSpec(type=DataType.long, name="foo"),
            ColSpec(type=DataType.integer, name="bar"),
        ]
    )
    dataset2 = MetaDataset(source=http_source, schema=schema)

    assert dataset1.digest != dataset2.digest

    delta_source = DeltaDatasetSource("fake/path/to/delta")
    dataset3 = MetaDataset(source=delta_source)
    assert dataset1.digest != dataset3.digest


def test_meta_dataset_with_uc_source():
    path = "/Volumes/dummy_catalog/dummy_schema/dummy_volume/tmp.yaml"

    with patch(
        "mlflow.data.uc_volume_dataset_source.UCVolumeDatasetSource._verify_uc_path_is_valid",
        side_effect=MlflowException(f"{path} does not exist in Databricks Unified Catalog."),
    ), pytest.raises(
        MlflowException, match=f"{path} does not exist in Databricks Unified Catalog."
    ):
        uc_volume_source = UCVolumeDatasetSource(path)

    with patch(
        "mlflow.data.uc_volume_dataset_source.UCVolumeDatasetSource._verify_uc_path_is_valid",
    ):
        uc_volume_source = UCVolumeDatasetSource(path)
        dataset = MetaDataset(source=uc_volume_source)
        json_str = dataset.to_json()
        parsed_json = json.loads(json_str)

        assert parsed_json["digest"] is not None
        assert path in parsed_json["source"]
        assert parsed_json["source_type"] == "uc_volume"
