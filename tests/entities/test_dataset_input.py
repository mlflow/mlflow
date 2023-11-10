from mlflow.entities import Dataset, DatasetInput, InputTag


def _check(dataset_input, tags, dataset):
    assert isinstance(dataset_input, DatasetInput)
    assert dataset_input.tags == tags
    assert dataset_input.dataset == dataset


def test_creation_and_hydration():
    key = "my_key"
    value = "my_value"
    tags = [InputTag(key, value)]
    name = "my_name"
    digest = "my_digest"
    source_type = "my_source_type"
    source = "my_source"
    schema = "my_schema"
    profile = "my_profile"
    dataset = Dataset(name, digest, source_type, source, schema, profile)
    dataset_input = DatasetInput(dataset=dataset, tags=tags)
    _check(dataset_input, tags, dataset)

    as_dict = {"dataset": dataset, "tags": tags}
    assert dict(dataset_input) == as_dict

    proto = dataset_input.to_proto()
    dataset_input2 = DatasetInput.from_proto(proto)
    _check(dataset_input2, tags, dataset)

    dataset_input3 = DatasetInput.from_dictionary(as_dict)
    _check(dataset_input3, tags, dataset)
