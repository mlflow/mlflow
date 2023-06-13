from mlflow.entities import Dataset


def _check(dataset, name, digest, source_type, source, schema=None, profile=None):
    assert isinstance(dataset, Dataset)
    assert dataset.name == name
    assert dataset.digest == digest
    assert dataset.source_type == source_type
    assert dataset.source == source
    assert dataset.schema == schema
    assert dataset.profile == profile


def test_creation_and_hydration():
    name = "my_name"
    digest = "my_digest"
    source_type = "my_source_type"
    source = "my_source"
    schema = "my_schema"
    profile = "my_profile"
    dataset = Dataset(name, digest, source_type, source, schema, profile)
    _check(dataset, name, digest, source_type, source, schema, profile)

    as_dict = {
        "name": name,
        "digest": digest,
        "source_type": source_type,
        "source": source,
        "schema": schema,
        "profile": profile,
    }
    assert dict(dataset) == as_dict

    proto = dataset.to_proto()
    dataset2 = Dataset.from_proto(proto)
    _check(dataset2, name, digest, source_type, source, schema, profile)

    dataset3 = Dataset.from_dictionary(as_dict)
    _check(dataset3, name, digest, source_type, source, schema, profile)
