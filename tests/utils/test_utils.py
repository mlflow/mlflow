from unittest import mock
import pytest

from mlflow.utils import (
    get_unique_resource_id,
    _chunk_dict,
    _truncate_dict,
    _get_fully_qualified_class_name,
)


def test_get_unique_resource_id_respects_max_length():
    for max_length in range(5, 30, 5):
        for _ in range(10000):
            assert len(get_unique_resource_id(max_length=max_length)) <= max_length


def test_get_unique_resource_id_with_invalid_max_length_throws_exception():
    with pytest.raises(ValueError):
        get_unique_resource_id(max_length=-50)

    with pytest.raises(ValueError):
        get_unique_resource_id(max_length=0)


def test_truncate_dict():
    d = {"12345": "12345"}
    length = 5

    with mock.patch("mlflow.utils._logger.warning") as mock_warning:
        max_legnth = length - 1

        # Truncate keys
        assert _truncate_dict(d, max_key_length=max_legnth) == {"1...": "12345"}
        mock_warning.assert_called_once_with("Truncated the key `1...`")
        mock_warning.reset_mock()

        # Truncate values
        assert _truncate_dict(d, max_value_length=max_legnth) == {"12345": "1..."}
        mock_warning.assert_called_once_with(
            "Truncated the value of the key `12345`. Truncated value: `1...`"
        )
        mock_warning.reset_mock()

        # Truncate both keys and values
        assert _truncate_dict(d, max_key_length=max_legnth, max_value_length=max_legnth) == {
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
    cap_name = None

    def f1(a1):
        nonlocal cap_name
        cap_name = _inspect_original_var_name(a1, 'unknown')

    xyz1 = object()
    f1(xyz1)
    assert cap_name == 'xyz1'

    cap_name = None

    f1(str(xyz1))
    assert cap_name == 'unknown'

    cap_name = None

    def f2(b1):
        f1(b1)

    f2(xyz1)
    assert cap_name == 'xyz1'

    cap_name_pos0 = None
    cap_name_kw_a1 = None

    def f3(a1, *, b1):
        nonlocal cap_name_pos0
        nonlocal cap_name_kw_a1
        cap_name_pos0 = _inspect_original_var_name(a1, None)
        cap_name_kw_a1 = _inspect_original_var_name(b1, None)

    xyz2 = object()
    xyz3 = object()

    f3(*[xyz2], **{'b1': xyz3})

    assert cap_name_pos0 == 'xyz2'
    assert cap_name_kw_a1 == 'xyz3'

