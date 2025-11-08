from unittest import mock

import pytest

from mlflow.utils import (
    AttrDict,
    _chunk_dict,
    _get_fully_qualified_class_name,
    _truncate_dict,
    merge_dicts,
)


def test_truncate_dict():
    d = {"12345": "12345"}
    length = 5

    with mock.patch("mlflow.utils._logger.warning") as mock_warning:
        max_length = length - 1

        # Truncate keys
        assert _truncate_dict(d, max_key_length=max_length) == {"1...": "12345"}
        mock_warning.assert_called_once_with("Truncated the key `1...`")
        mock_warning.reset_mock()

        # Truncate values
        assert _truncate_dict(d, max_value_length=max_length) == {"12345": "1..."}
        mock_warning.assert_called_once_with(
            "Truncated the value of the key `12345`. Truncated value: `1...`"
        )
        mock_warning.reset_mock()

        # Truncate both keys and values
        assert _truncate_dict(d, max_key_length=max_length, max_value_length=max_length) == {
            "1...": "1..."
        }
        assert mock_warning.call_count == 2
        (args1, _), (args2, _) = mock_warning.call_args_list
        assert args1[0] == "Truncated the key `1...`"
        assert args2[0] == "Truncated the value of the key `1...`. Truncated value: `1...`"

    assert _truncate_dict(d, max_key_length=length, max_value_length=length) == {"12345": "12345"}
    assert _truncate_dict(d, max_key_length=length + 1, max_value_length=length + 1) == {
        "12345": "12345"
    }

    with pytest.raises(
        ValueError, match="Must specify at least either `max_key_length` or `max_value_length`"
    ):
        _truncate_dict(d)


def test_merge_dicts():
    dict_a = {"a": 3, "b": {"c": {"d": [1, 2, 3]}}, "k": "hello"}
    dict_b = {"test_var": [1, 2]}
    expected_ab = {"a": 3, "b": {"c": {"d": [1, 2, 3]}}, "k": "hello", "test_var": [1, 2]}
    assert merge_dicts(dict_a, dict_b) == expected_ab

    dict_c = {"a": 10}
    with pytest.raises(ValueError, match="contains duplicate keys"):
        merge_dicts(dict_a, dict_c)

    expected_ac = {"a": 10, "b": {"c": {"d": [1, 2, 3]}}, "k": "hello"}
    assert merge_dicts(dict_a, dict_c, raise_on_duplicates=False) == expected_ac


def test_chunk_dict():
    d = {i: i for i in range(10)}
    assert list(_chunk_dict(d, 4)) == [
        {i: i for i in range(4)},
        {i: i for i in range(4, 8)},
        {i: i for i in range(8, 10)},
    ]
    assert list(_chunk_dict(d, 5)) == [
        {i: i for i in range(5)},
        {i: i for i in range(5, 10)},
    ]
    assert list(_chunk_dict(d, len(d))) == [d]
    assert list(_chunk_dict(d, len(d) + 1)) == [d]


def test_get_fully_qualified_class_name():
    class Foo:
        pass

    assert _get_fully_qualified_class_name(Foo()) == f"{__name__}.Foo"


def test_inspect_original_var_name():
    from mlflow.utils import _inspect_original_var_name

    def f1(a1, expected_name):
        assert _inspect_original_var_name(a1, "unknown") == expected_name

    xyz1 = object()
    f1(xyz1, "xyz1")

    f1(str(xyz1), "unknown")

    f1(None, "unknown")

    def f2(b1, expected_name):
        f1(b1, expected_name)

    f2(xyz1, "xyz1")

    def f3(a1, *, b1, expected_a1_name, expected_b1_name):
        assert _inspect_original_var_name(a1, None) == expected_a1_name
        assert _inspect_original_var_name(b1, None) == expected_b1_name

    xyz2 = object()
    xyz3 = object()

    f3(*[xyz2], **{"b1": xyz3, "expected_a1_name": "xyz2", "expected_b1_name": "xyz3"})


def test_random_name_generation():
    from mlflow.utils import name_utils

    # Validate exhausted loop truncation
    name = name_utils._generate_random_name(max_length=8)
    assert len(name) == 8

    # Validate default behavior while calling 1000 times that names end in integer
    names = [name_utils._generate_random_name() for i in range(1000)]
    assert all(len(name) <= 20 for name in names)
    assert all(name[-1].isnumeric() for name in names)


def test_basic_attribute_access():
    d = AttrDict({"a": 1, "b": 2})
    assert d.a == 1
    assert d.b == 2


def test_nested_attribute_access():
    d = AttrDict({"a": 1, "b": {"c": 3, "d": 4}})
    assert d.b.c == 3
    assert d.b.d == 4


def test_non_existent_attribute():
    d = AttrDict({"a": 1, "b": 2})
    with pytest.raises(AttributeError, match="'AttrDict' object has no attribute 'c'"):
        _ = d.c


def test_hasattr():
    d = AttrDict({"a": 1, "b": {"c": 3, "d": 4}})
    assert hasattr(d, "a")
    assert hasattr(d, "b")
    assert not hasattr(d, "e")
    assert hasattr(d.b, "c")
    assert not hasattr(d.b, "e")


def test_subclass_hasattr():
    class SubAttrDict(AttrDict):
        pass

    d = SubAttrDict({"a": 1, "b": {"c": 3, "d": 4}})
    assert hasattr(d, "a")
    assert not hasattr(d, "e")
    assert hasattr(d.b, "c")
    assert not hasattr(d.b, "e")

    with pytest.raises(AttributeError, match="'SubAttrDict' object has no attribute 'g'"):
        _ = d.g


def test_setattr():
    """Test that AttrDict supports setting attributes."""
    d = AttrDict({"a": 1, "b": 2})

    # Set existing attribute
    d.a = 10
    assert d.a == 10
    assert d["a"] == 10

    # Set new attribute
    d.c = 3
    assert d.c == 3
    assert d["c"] == 3
    assert "c" in d


def test_delattr():
    """Test that AttrDict supports deleting attributes."""
    d = AttrDict({"a": 1, "b": 2, "c": 3})

    # Delete existing attribute
    del d.b
    assert "b" not in d
    assert not hasattr(d, "b")

    # Verify other attributes still exist
    assert d.a == 1
    assert d.c == 3


def test_delattr_non_existent():
    d = AttrDict({"a": 1, "b": 2, "c": 3})
    with pytest.raises(KeyError, match="nonexistent"):
        del d.nonexistent
