import json

from mlflow.data.dataset import Dataset
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.types.schema import Schema

from tests.resources.data.dataset import SampleDataset
from tests.resources.data.dataset_source import SampleDatasetSource


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = SampleDataset(data_list=[1, 2, 3], source=source, name="testname")

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
    source = SampleDatasetSource._resolve(source_uri)
    dataset = SampleDataset(data_list=[1, 2, 3], source=source, name="testname")
    assert dataset.digest == dataset._compute_digest()


def test_expected_name_is_used():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    dataset_without_name = SampleDataset(data_list=[1, 2, 3], source=source)
    assert dataset_without_name.name == "dataset"

    dataset_with_name = SampleDataset(data_list=[1, 2, 3], source=source, name="testname")
    assert dataset_with_name.name == "testname"


def test_create_dataset_from_only_source():
    source_uri = "test:/my/test/uri"
    source = HTTPDatasetSource(url=source_uri)
    dataset = Dataset(source=source)

    json_str = dataset.to_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["digest"] != None
    assert parsed_json["source"] == '{"url": "test:/my/test/uri"}'
    assert parsed_json["source_type"] == "http"
