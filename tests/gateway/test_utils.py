import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.utils import (
    join_url,
    is_valid_endpoint_name,
    check_configuration_route_name_collisions,
    _is_valid_uri,
    assemble_uri_path,
    set_gateway_uri,
    get_gateway_uri,
    SearchRoutesToken,
)
import mlflow.gateway.envs


@pytest.mark.parametrize(
    "base_url, route",
    [
        ("http://127.0.0.1:6000", "/api/2.0/gateway/routes/test"),
        ("http://127.0.0.1:6000/", "/api/2.0/gateway/routes/test"),
        ("http://127.0.0.1:6000/api/2.0/gateway", "/routes/test"),
        ("http://127.0.0.1:6000/api/2.0/gateway/", "/routes/test"),
        ("http://127.0.0.1:6000", "api/2.0/gateway/routes/test"),
        ("http://127.0.0.1:6000/", "api/2.0/gateway/routes/test"),
        ("http://127.0.0.1:6000/api/2.0/gateway", "routes/test"),
        ("http://127.0.0.1:6000/api/2.0/gateway/", "routes/test"),
    ],
)
def test_join_url(base_url, route):
    assert join_url(base_url, route) == "http://127.0.0.1:6000/api/2.0/gateway/routes/test"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("validName", True),
        ("invalid name", False),
        ("invalid/name", False),
        ("invalid?name", False),
        ("", False),
    ],
)
def test_is_valid_endpoint_name(name, expected):
    assert is_valid_endpoint_name(name) == expected


def test_check_configuration_route_name_collisions():
    config = {"routes": [{"name": "name1"}, {"name": "name2"}, {"name": "name1"}]}
    with pytest.raises(MlflowException, match="Duplicate names found in route configurations"):
        check_configuration_route_name_collisions(config)


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("http://localhost", True),
        ("databricks", True),
        ("localhost", False),
        ("http:/localhost", False),
        ("", False),
    ],
)
def test__is_valid_uri(uri, expected):
    assert _is_valid_uri(uri) == expected


@pytest.mark.parametrize(
    "paths, expected",
    [
        (["path1", "path2", "path3"], "/path1/path2/path3"),
        (["/path1/", "/path2/", "/path3/"], "/path1/path2/path3"),
        (["/path1//", "/path2//", "/path3//"], "/path1/path2/path3"),
        (["path1", "", "path3"], "/path1/path3"),
        (["", "", ""], "/"),
        ([], "/"),
    ],
)
def test_assemble_uri_path(paths, expected):
    assert assemble_uri_path(paths) == expected


def test_set_gateway_uri(monkeypatch):
    monkeypatch.setattr("mlflow.gateway.utils._gateway_uri", None)

    valid_uri = "http://localhost"
    set_gateway_uri(valid_uri)
    assert mlflow.gateway.utils._gateway_uri == valid_uri

    invalid_uri = "localhost"
    with pytest.raises(MlflowException, match="The gateway uri provided is missing required"):
        set_gateway_uri(invalid_uri)


def test_get_gateway_uri(monkeypatch):
    monkeypatch.setattr("mlflow.gateway.utils._gateway_uri", None)
    monkeypatch.setattr("mlflow.gateway.envs.MLFLOW_GATEWAY_URI.get", lambda: None)

    with pytest.raises(MlflowException, match="No Gateway server uri has been set"):
        get_gateway_uri()

    valid_uri = "http://localhost"
    monkeypatch.setattr("mlflow.gateway.utils._gateway_uri", valid_uri)
    assert get_gateway_uri() == valid_uri

    monkeypatch.setattr("mlflow.gateway.utils._gateway_uri", None)
    monkeypatch.setattr("mlflow.gateway.envs.MLFLOW_GATEWAY_URI.get", lambda: valid_uri)
    assert get_gateway_uri() == valid_uri


@pytest.mark.parametrize(
    "index, should_raise",
    [
        (123, False),
        ("not an integer", True),
        (-1, True),
        (None, True),
        ([1, 2, 3], True),
        ({"key": "value"}, True),
    ],
)
def test_search_routes_token(index, should_raise):
    token = SearchRoutesToken(index)
    encoded_token = token.encode()
    if should_raise:
        with pytest.raises(MlflowException, match="Invalid SearchRoutes token"):
            SearchRoutesToken.decode(encoded_token)
    else:
        decoded_token = SearchRoutesToken.decode(encoded_token)
        assert decoded_token.index == token.index
