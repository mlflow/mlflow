import json

from mlflow.types.schema import Schema

from tests.resources.data.dataset_source import TestDatasetSource
from tests.resources.data.dataset import TestDataset


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    dataset = TestDataset(data_list=[1, 2, 3], source=source, name="testname")

    dataset_json = dataset.to_json()
    parsed_json = json.loads(dataset_json)
    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "size"}
    assert parsed_json["name"] == dataset.name
    assert parsed_json["digest"] == dataset.digest
    assert parsed_json["source"] == dataset.source.to_json()
    assert parsed_json["source_type"] == dataset.source._get_source_type()
    assert parsed_json["size"] == json.dumps(dataset.size)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == dataset.schema


def test_digest_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    dataset = TestDataset(data_list=[1, 2, 3], source=source, name="testname")
    assert dataset.digest == dataset._compute_digest()
