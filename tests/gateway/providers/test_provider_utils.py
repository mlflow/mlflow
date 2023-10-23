import pytest

from mlflow.gateway.providers.utils import (
    dict_contains_nested_path,
    get_dict_value_by_path,
    rename_payload_keys,
)


def test_rename_payload_keys():
    payload = {"old_key1": "value1", "old_key2": "value2", "old_key3": None, "old_key4": []}
    mapping = {"old_key1": "new_key1", "old_key2": "new_key2"}
    expected = {"new_key1": "value1", "new_key2": "value2", "old_key3": None, "old_key4": []}
    assert rename_payload_keys(payload, mapping) == expected


@pytest.mark.parametrize(
    ("payload", "mapping", "expected"),
    [
        (
            {"old_key1": "value1", "old_key2": None, "old_key3": "value3"},
            {"old_key1": "new_key1", "old_key3": "new_key3"},
            {"new_key1": "value1", "old_key2": None, "new_key3": "value3"},
        ),
        (
            {"old_key1": None, "old_key2": "value2", "old_key3": []},
            {"old_key1": "new_key1", "old_key3": "new_key3"},
            {"new_key1": None, "old_key2": "value2", "new_key3": []},
        ),
        (
            {"old_key1": "value1", "old_key2": "value2"},
            {"old_key1": "new_key1", "old_key3": "new_key3"},
            {"new_key1": "value1", "old_key2": "value2"},
        ),
        (
            {"old_key1": "value1", "old_key2": "value2", "old_key3": "value3"},
            {"old_key1": "new_key.key1", "old_key2": "new_key.key2"},
            {"new_key": {"key1": "value1", "key2": "value2"}, "old_key3": "value3"},
        ),
    ],
)
def test_rename_payload_keys_parametrized(payload, mapping, expected):
    assert rename_payload_keys(payload, mapping) == expected


@pytest.mark.parametrize(
    ("payload", "path", "expected"), [({"a": 1, "b": 2}, "a", 1), ({"a": {"b": 1}}, "a.b", 1)]
)
def test_get_dict_value_by_path(payload, path, expected):
    value = get_dict_value_by_path(payload, path)
    assert value == expected


def test_get_dict_value_by_path_fails_with_keyerror():
    with pytest.raises(KeyError, match=r".*") as e:
        get_dict_value_by_path({"a": 1}, "b")
    assert e is not None


@pytest.mark.parametrize(
    ("payload", "path", "expected"),
    [
        ({"a": 1, "b": 2}, "a", True),
        ({"a": {"b": 1}}, "a.b", True),
        ({"a": 1}, "b", False),
        ({"a": {"b": 1}}, "a.c", False),
        ({"a": {"b": 1}}, "x.y", False),
    ],
)
def test_dict_contains_nested_key(payload, path, expected):
    assert dict_contains_nested_path(payload, path) == expected
