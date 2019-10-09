import pytest

from mlflow.utils import get_unique_resource_id


def test_get_unique_resource_id_respects_max_length():
    for max_length in range(5, 30, 5):
        for _ in range(10000):
            assert len(get_unique_resource_id(max_length=max_length)) <= max_length


def test_get_unique_resource_id_with_invalid_max_length_throws_exception():
    with pytest.raises(ValueError):
        get_unique_resource_id(max_length=-50)

    with pytest.raises(ValueError):
        get_unique_resource_id(max_length=0)
