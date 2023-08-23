import pytest

from mlflow.gateway.providers.utils import (
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
    ],
)
def test_rename_payload_keys_parametrized(payload, mapping, expected):
    assert rename_payload_keys(payload, mapping) == expected
