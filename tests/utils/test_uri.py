import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils import extract_db_type_from_uri
from mlflow.utils.uri import _is_databricks_uri, _is_http_uri, _is_local_uri, \
    get_db_profile_from_uri, get_uri_scheme


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


def test_get_db_profile_from_uri_casing():
    assert get_db_profile_from_uri('databricks://aAbB') == 'aAbB'


def test_uri_types():
    assert _is_local_uri("mlruns")
    assert _is_local_uri("./mlruns")
    assert _is_local_uri("file:///foo/mlruns")
    assert _is_local_uri("file:foo/mlruns")
    assert not _is_local_uri("https://whatever")
    assert not _is_local_uri("http://whatever")
    assert not _is_local_uri("databricks")
    assert not _is_local_uri("databricks:whatever")
    assert not _is_local_uri("databricks://whatever")

    assert _is_databricks_uri("databricks")
    assert _is_databricks_uri("databricks:whatever")
    assert _is_databricks_uri("databricks://whatever")
    assert not _is_databricks_uri("mlruns")
    assert not _is_databricks_uri("http://whatever")

    assert _is_http_uri("http://whatever")
    assert _is_http_uri("https://whatever")
    assert not _is_http_uri("file://whatever")
    assert not _is_http_uri("databricks://whatever")
    assert not _is_http_uri("mlruns")
