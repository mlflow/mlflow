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
    handle_incomplete_chunks,
    is_valid_endpoint_name,
    parse_sse_line,
    resolve_route_url,
    set_gateway_uri,
    stream_sse_data,
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


def test_parse_sse_line_single_data_line():
    line = b'data: {"message": "hello"}'
    assert parse_sse_line(line) == {"message": "hello"}


def test_parse_sse_line_with_newline():
    line = b'data: {"message": "hello"}\n'
    assert parse_sse_line(line) == {"message": "hello"}


def test_parse_sse_line_event_line():
    line = b"event: message"
    assert parse_sse_line(line) is None


def test_parse_sse_line_done_marker():
    line = b"data: [DONE]"
    assert parse_sse_line(line) is None


def test_parse_sse_line_empty_data():
    line = b"data: "
    assert parse_sse_line(line) is None


def test_parse_sse_line_empty():
    assert parse_sse_line(b"") is None


def test_parse_sse_line_string_input():
    line = 'data: {"key": "value"}'
    assert parse_sse_line(line) == {"key": "value"}


def test_parse_sse_line_invalid_json():
    line = b"data: {invalid json}"
    assert parse_sse_line(line) is None


def test_parse_sse_line_invalid_utf8():
    line = b"\xff\xfe"
    assert parse_sse_line(line) is None


def test_parse_sse_line_non_data_line_ignored():
    assert parse_sse_line(b"id: 123") is None
    assert parse_sse_line(b"retry: 1000") is None


@pytest.mark.asyncio
async def test_stream_sse_data_yields_parsed_json():
    async def mock_stream():
        yield b'data: {"chunk": 1}\n'
        yield b'data: {"chunk": 2}\n'

    results = [data async for data in stream_sse_data(mock_stream())]
    assert results == [{"chunk": 1}, {"chunk": 2}]


@pytest.mark.asyncio
async def test_stream_sse_data_skips_done():
    async def mock_stream():
        yield b'data: {"chunk": 1}\n'
        yield b"data: [DONE]\n"

    results = [data async for data in stream_sse_data(mock_stream())]
    assert results == [{"chunk": 1}]


@pytest.mark.asyncio
async def test_stream_sse_data_skips_empty_lines():
    async def mock_stream():
        yield b""
        yield b'data: {"chunk": 1}\n'
        yield b"   "
        yield b'data: {"chunk": 2}\n'

    results = [data async for data in stream_sse_data(mock_stream())]
    assert results == [{"chunk": 1}, {"chunk": 2}]


@pytest.mark.asyncio
async def test_stream_sse_data_skips_invalid_json():
    async def mock_stream():
        yield b'data: {"valid": true}\n'
        yield b"data: not json\n"
        yield b'data: {"also_valid": true}\n'

    results = [data async for data in stream_sse_data(mock_stream())]
    assert results == [{"valid": True}, {"also_valid": True}]


@pytest.mark.asyncio
async def test_handle_incomplete_chunks_complete_lines():
    async def mock_stream():
        yield b"line1\nline2\n"

    results = [chunk async for chunk in handle_incomplete_chunks(mock_stream())]
    assert results == [b"line1", b"line2"]


@pytest.mark.asyncio
async def test_handle_incomplete_chunks_split_across_chunks():
    async def mock_stream():
        yield b"he"
        yield b"llo\nwor"
        yield b"ld\n"

    results = [chunk async for chunk in handle_incomplete_chunks(mock_stream())]
    assert results == [b"hello", b"world"]


@pytest.mark.asyncio
async def test_handle_incomplete_chunks_trailing_data():
    async def mock_stream():
        yield b"line1\nline2"

    results = [chunk async for chunk in handle_incomplete_chunks(mock_stream())]
    assert results == [b"line1", b"line2"]


@pytest.mark.asyncio
async def test_handle_incomplete_chunks_no_newline():
    async def mock_stream():
        yield b"no newline at all"

    results = [chunk async for chunk in handle_incomplete_chunks(mock_stream())]
    assert results == [b"no newline at all"]
