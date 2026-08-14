from mlflow.entities import InputTag


def _check(input_tag, key, value):
    assert isinstance(input_tag, InputTag)
    assert input_tag.key == key
    assert input_tag.value == value


def test_creation_and_hydration():
    key = "my_key"
    value = "my_value"
    input_tag = InputTag(key, value)
    _check(input_tag, key, value)

    as_dict = {
        "key": key,
        "value": value,
    }
    assert dict(input_tag) == as_dict

    proto = input_tag.to_proto()
    input_tag2 = InputTag.from_proto(proto)
    _check(input_tag2, key, value)

    input_tag3 = InputTag.from_dictionary(as_dict)
    _check(input_tag3, key, value)
