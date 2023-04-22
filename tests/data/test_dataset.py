import json

import pandas as pd
import numpy as np

import mlflow.data
from mlflow.types.schema import Schema
from mlflow.utils.name_utils import _generate_dataset_name

from tests.resources.data.dataset_source import TestDatasetSource
from tests.resources.data.dataset import TestDataset


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    dataset = TestDataset(data_list=[1, 2, 3], source=source, name="testname")

    dataset_json = dataset.to_json()
    parsed_json = json.loads(dataset_json)
    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert parsed_json["name"] == dataset.name
    assert parsed_json["digest"] == dataset.digest
    assert parsed_json["source"] == dataset.source.to_json()
    assert parsed_json["source_type"] == dataset.source._get_source_type()
    assert parsed_json["profile"] == json.dumps(dataset.profile)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == dataset.schema


def test_digest_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    dataset = TestDataset(data_list=[1, 2, 3], source=source, name="testname")
    assert dataset.digest == dataset._compute_digest()


def test_datasets_are_named_as_expected(tmp_path):
    """
    This test verifies that the logic used to generate a deterministic name for an MLflow Tracking
    dataset based on the dataset's hash does not change. Be **very** careful if editing this test,
    and do not change the deterministic mapping behavior of _generate_dataset_name, which is used
    by mlflow.data.Dataset to generate dataset names if users do not specify names for their
    datasets. Otherwise, user workflows may break.
    """
    df = pd.DataFrame.from_dict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    dataset1 = mlflow.data.from_pandas(df, source=tmp_path)
    assert dataset1.name == "diverse-data-99798" == _generate_dataset_name(dataset1.digest)

    arr = np.array([["cat", "dog", "moose"], ["cow", "bird", "lobster"]])
    dataset2 = mlflow.data.from_numpy(arr, source=tmp_path)
    assert dataset2.name == "learned-data-14980" == _generate_dataset_name(dataset2.digest)
