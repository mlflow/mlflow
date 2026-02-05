import pytest
from fastapi import HTTPException

from mlflow.exceptions import MlflowException
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.utils import (
    SearchRoutesToken,
    _is_valid_uri,
    assemble_uri_path,
    check_configuration_route_name_collisions,
    get_gateway_uri,
    is_valid_endpoint_name,
    resolve_route_url,
    set_gateway_uri,
    translate_http_exception,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@pytest.mark.parametrize(
    ("base_url", "route"),
    [
        ("http://127.0.0.1:6000", "gateway/test/invocations"),
        ("http://127.0.0.1:6000/", "/gateway/test/invocations"),
        ("http://127.0.0.1:6000/gateway", "/test/invocations"),
        ("http://127.0.0.1:6000/gateway/", "/test/invocations"),
        ("http://127.0.0.1:6000/gateway", "test/invocations"),
        ("http://127.0.0.1:6000/gateway/", "test/invocations"),
    ],
)
def test_resolve_route_url(base_url, route):
    assert resolve_route_url(base_url, route) == "http://127.0.0.1:6000/gateway/test/invocations"


@pytest.mark.parametrize("base_url", ["databricks", "databricks://my.workspace"])
def test_resolve_route_url_qualified_url_ignores_base(base_url):
    route = "https://my.databricks.workspace/api/2.0/gateway/chat/invocations"

    resolved = resolve_route_url(base_url, route)

    assert resolved == route


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("validName", True),
        ("valid-name", True),
        ("valid_name", True),
        ("valid.name", True),
        ("valid123", True),
        ("invalid name", False),
        ("invalid/name", False),
        ("invalid?name", False),
        ("", False),
        ("日本語", False),  # Japanese characters
        ("naïve", False),  # accented characters
        ("名前", False),  # Chinese characters
    ],
)
def test_is_valid_endpoint_name(name, expected):
    assert is_valid_endpoint_name(name) == expected


def test_check_configuration_route_name_collisions():
    config = {"endpoints": [{"name": "name1"}, {"name": "name2"}, {"name": "name1"}]}
    with pytest.raises(
        MlflowException, match="Duplicate names found in endpoint / route configurations"
    ):
        check_configuration_route_name_collisions(config)


@pytest.mark.parametrize(
    ("uri", "expected"),
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
    ("paths", "expected"),
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
    assert get_gateway_uri() == valid_uri

    invalid_uri = "localhost"
    with pytest.raises(MlflowException, match="The gateway uri provided is missing required"):
        set_gateway_uri(invalid_uri)


def test_get_gateway_uri(monkeypatch):
    monkeypatch.setattr("mlflow.gateway.utils._gateway_uri", None)
    monkeypatch.delenv("MLFLOW_GATEWAY_URI", raising=False)

    with pytest.raises(MlflowException, match="No Gateway server uri has been set"):
        get_gateway_uri()

    valid_uri = "http://localhost"
    monkeypatch.setattr("mlflow.gateway.utils._gateway_uri", valid_uri)
    assert get_gateway_uri() == valid_uri

    monkeypatch.delenv("MLFLOW_GATEWAY_URI", raising=False)
    set_gateway_uri(valid_uri)
    assert get_gateway_uri() == valid_uri


def test_search_routes_token_decodes_correctly():
    token = SearchRoutesToken(12345)
    encoded_token = token.encode()
    decoded_token = SearchRoutesToken.decode(encoded_token)
    assert decoded_token.index == token.index


@pytest.mark.parametrize(
    "index",
    [
        "not an integer",
        -1,
        None,
        [1, 2, 3],
        {"key": "value"},
    ],
)
def test_search_routes_token_with_invalid_token_values(index):
    token = SearchRoutesToken(index)
    encoded_token = token.encode()
    with pytest.raises(MlflowException, match="Invalid SearchRoutes token"):
        SearchRoutesToken.decode(encoded_token)


@pytest.mark.asyncio
async def test_translate_http_exception_handles_ai_gateway_exception():
    @translate_http_exception
    async def raise_ai_gateway_exception():
        raise AIGatewayException(status_code=503, detail="AI Gateway error")

    with pytest.raises(HTTPException, match="AI Gateway error") as exc_info:
        await raise_ai_gateway_exception()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "AI Gateway error"


@pytest.mark.asyncio
async def test_translate_http_exception_handles_mlflow_exception():
    @translate_http_exception
    async def raise_mlflow_exception():
        raise MlflowException("Invalid parameter", error_code=INVALID_PARAMETER_VALUE)

    with pytest.raises(HTTPException, match="Invalid parameter") as exc_info:
        await raise_mlflow_exception()

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error_code": "INVALID_PARAMETER_VALUE",
        "message": "Invalid parameter",
    }


@pytest.mark.asyncio
async def test_translate_http_exception_passes_through_other_exceptions():
    @translate_http_exception
    async def raise_value_error():
        raise ValueError("Some value error")

    with pytest.raises(ValueError, match="Some value error"):
        await raise_value_error()
