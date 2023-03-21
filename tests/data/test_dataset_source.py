import json

from tests.resources.data.dataset_source import TestDatasetSource


def test_load(tmp_path):
    assert TestDatasetSource("test:" + str(tmp_path)).load() == str(tmp_path)


def test_conversion_to_json_and_back():
    uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(uri)
    source_json = source.to_json()
    assert json.loads(source_json)["uri"] == uri
    reloaded_source = TestDatasetSource.from_json(source_json)
    assert reloaded_source.uri == source.uri
