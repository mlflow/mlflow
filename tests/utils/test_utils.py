import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.dbmodels.db_types import DATABASE_ENGINES
from mlflow.utils import get_unique_resource_id, extract_db_type_from_uri, get_uri_scheme


def test_get_unique_resource_id_respects_max_length():
    for max_length in range(5, 30, 5):
        for _ in range(10000):
            assert len(get_unique_resource_id(max_length=max_length)) <= max_length


def test_get_unique_resource_id_with_invalid_max_length_throws_exception():
    with pytest.raises(ValueError):
        get_unique_resource_id(max_length=-50)

    with pytest.raises(ValueError):
        get_unique_resource_id(max_length=0)


def test_extract_db_type_from_uri():
    uri = "{}://username:password@host:port/database"
    for legit_db in DATABASE_ENGINES:
        assert legit_db == extract_db_type_from_uri(uri.format(legit_db))
        assert legit_db == get_uri_scheme(uri.format(legit_db))

        with_driver = legit_db + "+driver-string"
        assert legit_db == extract_db_type_from_uri(uri.format(with_driver))
        assert legit_db == get_uri_scheme(uri.format(with_driver))

    for unsupported_db in ["a", "aa", "sql"]:
        with pytest.raises(MlflowException):
            extract_db_type_from_uri(unsupported_db)
