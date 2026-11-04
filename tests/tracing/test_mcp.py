import json
import os
import select
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from threading import Barrier, Event
from unittest import mock

import pytest
import requests

import mlflow
from mlflow.entities import MCPAccessEndpoint, MCPServerRef, MCPServerVersion, NoOpSpan, TraceData
from mlflow.entities.mcp_server import MCPRemoteTransportType
from mlflow.entities.trace_mcp_server_link import TraceMCPServerLink
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.constant import SpanAttributeKey, TraceTagKey
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.utils import mcp
from mlflow.tracing.utils.mcp import _resolution_cache, resolve_mcp_server
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.workspace_context import (
    ServerWorkspaceContext,
    WorkspaceContext,
    get_request_workspace,
)
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

from tests.tracing.helper import skip_when_testing_trace_sdk

NAME = "com.example/weather"
URI = "http://tracking.example.com"


def _version(version="1.0.0", workspace="default", name=NAME):
    return MCPServerVersion(name=name, version=version, server_json={}, workspace=workspace)


def _endpoint(**kwargs):
    return MCPAccessEndpoint(**{
        "id": "ae-weather",
        "server_name": NAME,
        "url": "https://example.com/mcp",
        "workspace": "default",
        **kwargs,
    })


def _resolve(server, *, workspace="default", tracking_uri=URI, cache_ttl_seconds=None):
    return resolve_mcp_server(
        server=server,
        tracking_uri=tracking_uri,
        workspace=workspace,
        cache_ttl_seconds=cache_ttl_seconds,
    )


@pytest.fixture(autouse=True)
def clear_mcp_cache(monkeypatch):
    monkeypatch.delenv("MLFLOW_MCP_SERVER_CACHE_TTL_SECONDS", raising=False)
    _resolution_cache.clear()
    yield
    _resolution_cache.clear()


@pytest.fixture
def registry_store():
    with mock.patch(
        "mlflow.tracing.utils.mcp._get_store",
        return_value=mock.MagicMock(supports_workspaces=True),
    ) as get_store:
        yield get_store.return_value
        get_store.assert_called()


@pytest.mark.parametrize(
    ("server", "version"),
    [
        (MCPServerRef(NAME), None),
        (MCPServerRef(NAME, version="latest"), "latest"),
        (_version(), "1.0.0"),
    ],
)
@pytest.mark.parametrize("tool", [None, "forecast"])
def test_link_direct_identity_without_lookup(server, version, tool):
    with mock.patch("mlflow.tracing.utils.mcp._get_store") as client:
        with mlflow.start_span("mcp") as span:
            span.link_mcp_server(server, tool=tool)
            expected = {SpanAttributeKey.MCP_SERVER_NAME: NAME}
            if version is not None:
                expected[SpanAttributeKey.MCP_SERVER_VERSION] = version
            if tool is not None:
                expected[SpanAttributeKey.MCP_TOOL_NAME] = tool
            assert {
                k: v for k, v in span.attributes.items() if k.startswith("mlflow.mcp.")
            } == expected
        client.assert_not_called()


def test_link_mcp_server_rejects_ended_span():
    with mlflow.start_span("mcp") as span:
        pass
    with pytest.raises(
        MlflowException,
        match=r"Cannot link an MCP server to a span that is no longer recording\.",
    ) as exc_info:
        span.link_mcp_server(MCPServerRef(NAME))
    assert str(exc_info.value) == "Cannot link an MCP server to a span that is no longer recording."


@pytest.mark.parametrize(
    "second",
    [
        MCPServerRef("com.example/other", version="1.0.0"),
        MCPServerRef(NAME),
        MCPServerRef(NAME, version="2.0.0"),
    ],
)
def test_conflicting_identity_does_not_change_attributes(second):
    with mlflow.start_span("mcp") as span:
        span.link_mcp_server(_version(), tool="forecast")
        before = span.attributes.copy()
        with pytest.raises(MlflowException, match="only one MCP server identity"):
            span.link_mcp_server(second, tool="different")
        assert span.attributes == before


def test_repeat_identity_and_existing_otel_attributes():
    with mlflow.start_span("mcp") as span:
        span._span.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, NAME)
        span._span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, "1.0.0")
        span.link_mcp_server(MCPServerRef(NAME, version="1.0.0"), tool="forecast")
        span.link_mcp_server(_version())
        assert span.get_attribute(SpanAttributeKey.MCP_TOOL_NAME) == "forecast"
        with pytest.raises(MlflowException, match="only one MCP server identity"):
            span.link_mcp_server(MCPServerRef(NAME))


@pytest.mark.parametrize("version", ["1", "1.0", "true", "null", "[]", "{}"])
@pytest.mark.parametrize("same_identity", [False, True])
def test_existing_otel_version_preserves_string_identity(version, same_identity):
    with mlflow.start_span("mcp") as span:
        span._span.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, NAME)
        span._span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, version)
        if same_identity:
            span.link_mcp_server(MCPServerRef(NAME, version=version))
            assert span.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == version
        else:
            before = dict(span._span.attributes)
            with pytest.raises(MlflowException, match="only one MCP server identity"):
                span.link_mcp_server(MCPServerRef(NAME))
            assert dict(span._span.attributes) == before


def test_concurrent_conflicting_links_keep_one_complete_identity():
    barrier = Barrier(2, timeout=10)

    def resolve(server, **kwargs):
        barrier.wait()
        return resolve_mcp_server(server=server, **kwargs)

    with mlflow.start_span("mcp") as span:
        servers = [MCPServerRef(NAME, version="1"), MCPServerRef("com.example/other", version="2")]

        def link(server):
            try:
                span.link_mcp_server(server, tool=server.name)
            except MlflowException as error:
                return error
            return server

        with (
            mock.patch(
                "mlflow.tracing.utils.mcp.resolve_mcp_server", side_effect=resolve
            ) as resolver,
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-mcp-link") as executor,
        ):
            results = list(executor.map(link, servers))
        winner = next(result for result in results if isinstance(result, MCPServerRef))
        failures = [result for result in results if isinstance(result, MlflowException)]
        assert len(failures) == 1
        assert "only one MCP server identity" in str(failures[0])
        assert span.get_attribute(SpanAttributeKey.MCP_SERVER_NAME) == winner.name
        assert span.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == winner.version
        assert span.get_attribute(SpanAttributeKey.MCP_TOOL_NAME) == winner.name
        assert resolver.call_count == 2


@pytest.mark.parametrize("server", [NAME, "https://example.com/mcp", None, {}])
def test_invalid_server(server):
    with mlflow.start_span("mcp") as span:
        before = span.attributes.copy()
        with pytest.raises(MlflowException, match="server must be"):
            span.link_mcp_server(server)
        assert span.attributes == before


@pytest.mark.parametrize("tool", ["", " ", 1, "high\ud800", "low\udfff"])
def test_invalid_tool_before_lookup(tool):
    with mock.patch("mlflow.tracing.utils.mcp._get_store") as client:
        with mlflow.start_span("mcp") as span:
            before = span.attributes.copy()
            with pytest.raises(MlflowException, match="tool name"):
                span.link_mcp_server(MCPServerRef(NAME, alias="prod"), tool=tool)
            assert span.attributes == before
        client.assert_not_called()


def test_noop_span_does_not_resolve():
    with mock.patch("mlflow.tracing.utils.mcp._get_store") as client:
        assert NoOpSpan().link_mcp_server(MCPServerRef(NAME, alias="prod")) is None
        client.assert_not_called()


@skip_when_testing_trace_sdk
@pytest.mark.parametrize(
    "server",
    [MCPServerRef(NAME, alias="prod"), _endpoint()],
    ids=["alias", "endpoint"],
)
def test_file_store_mcp_resolution_fails_with_actionable_error(server, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()

    with pytest.raises(MlflowException, match="FileStore backend does not support these lookups"):
        _resolve(server, tracking_uri=tracking_uri, cache_ttl_seconds=0)


def test_alias_resolution_records_concrete_version_and_preserves_history(registry_store):
    with mock.patch.object(
        registry_store,
        "get_mcp_server_version_by_alias",
        side_effect=[_version(), _version("2.0.0")],
    ) as lookup:
        with mlflow.start_span("first") as first:
            first.link_mcp_server(MCPServerRef(NAME, alias="prod"), cache_ttl_seconds=0)
        with mlflow.start_span("second") as second:
            second.link_mcp_server(MCPServerRef(NAME, alias="prod"), cache_ttl_seconds=0)
        assert first.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == "1.0.0"
        assert second.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == "2.0.0"
        assert lookup.call_args_list == [mock.call(name=NAME, alias="prod")] * 2


@pytest.mark.parametrize("endpoint", [False, True])
def test_resolution_failure_is_not_cached_or_partially_written(endpoint, registry_store):
    server = _endpoint() if endpoint else MCPServerRef(NAME, alias="missing")
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
    with mock.patch.object(
        registry_store, method, side_effect=MlflowException("not found")
    ) as lookup:
        with mlflow.start_span("mcp") as span:
            before = span.attributes.copy()
            for _ in range(2):
                with pytest.raises(MlflowException, match="not found"):
                    span.link_mcp_server(server, tool="forecast")
                assert span.attributes == before
        assert lookup.call_count == 2


@pytest.mark.parametrize("resolved_version", [None, _version("2.0.0")])
def test_endpoint_reloads_instead_of_using_stale_resolution(resolved_version, registry_store):
    endpoint = _endpoint(resolved_version=_version())
    with mock.patch.object(
        registry_store,
        "get_mcp_access_endpoint",
        return_value=replace(endpoint, resolved_version=resolved_version),
    ) as lookup:
        registry_store.attach_mock(lookup, "get_mcp_access_endpoint")
        with mlflow.start_span("endpoint") as span:
            span.link_mcp_server(endpoint)
            assert span.get_attribute(SpanAttributeKey.MCP_SERVER_NAME) == NAME
            if resolved_version is None:
                assert SpanAttributeKey.MCP_SERVER_VERSION not in span.attributes
            else:
                assert span.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == "2.0.0"
            assert SpanAttributeKey.MCP_TOOL_NAME not in span.attributes
        lookup.assert_called_once_with(server_name=NAME, endpoint_id=endpoint.id)
        assert registry_store.mock_calls == [
            mock.call.get_mcp_access_endpoint(server_name=NAME, endpoint_id=endpoint.id)
        ]


@pytest.mark.parametrize(
    "changed",
    [{"url": "https://example.com/new"}, {"transport_type": MCPRemoteTransportType.SSE}],
)
def test_endpoint_change_rejected_and_not_cached(changed, registry_store):
    endpoint = _endpoint()
    updated = replace(endpoint, **changed)
    with mock.patch.object(
        registry_store, "get_mcp_access_endpoint", return_value=updated
    ) as lookup:
        with pytest.raises(MlflowException, match="URL or transport has changed"):
            _resolve(endpoint)
        assert _resolve(updated).name == NAME
        assert lookup.call_count == 2
        with pytest.raises(MlflowException, match="URL or transport has changed"):
            _resolve(endpoint)
        assert lookup.call_count == 3


@pytest.mark.parametrize(
    "changed",
    [{"url": "https://example.com/new"}, {"transport_type": MCPRemoteTransportType.SSE}],
)
def test_reloaded_endpoint_refreshes_incompatible_cached_endpoint(changed, registry_store):
    endpoint = _endpoint(resolved_version=_version())
    updated = replace(endpoint, **changed, resolved_version=_version("2.0.0"))
    with mock.patch.object(
        registry_store, "get_mcp_access_endpoint", side_effect=[endpoint, updated]
    ) as lookup:
        assert _resolve(endpoint).version == "1.0.0"
        assert _resolve(updated).version == "2.0.0"
        assert _resolve(updated).version == "2.0.0"
        assert lookup.call_args_list == [mock.call(server_name=NAME, endpoint_id=endpoint.id)] * 2


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("include_workspace", [False, True])
@pytest.mark.parametrize(
    ("workspace", "supports_workspaces"),
    [(None, False), (None, True), ("default", True), ("team-a", True)],
)
def test_resolution_through_rest_store(
    workspace, supports_workspaces, include_workspace, endpoint, monkeypatch
):
    expected_workspace = workspace or "default"
    store = RestStore(lambda: MlflowHostCreds(URI))
    store._workspace_support = supports_workspaces
    version = {"name": NAME, "version": "1", "server_json": {}}
    if include_workspace:
        version["workspace"] = expected_workspace
    data = (
        {
            "id": "ae-weather",
            "server_name": NAME,
            "url": "https://example.com/mcp",
            "resolved_version": version,
        }
        if endpoint
        else version
    )
    if include_workspace:
        data["workspace"] = expected_workspace
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(data).encode()
    server = (
        _endpoint(workspace=expected_workspace) if endpoint else MCPServerRef(NAME, alias="prod")
    )
    with (
        mock.patch("mlflow.tracing.utils.mcp._get_store", return_value=store) as get_store,
        mock.patch("requests.Session.request", return_value=response) as request,
        ServerWorkspaceContext(workspace),
        monkeypatch.context() as workspace_env,
    ):
        with mlflow.start_span("mcp") as span:
            with ServerWorkspaceContext("different"):
                workspace_env.setenv("MLFLOW_WORKSPACE", "different")
                span.link_mcp_server(server)
                assert get_request_workspace() == "different"
            assert span.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == "1"
        get_store.assert_called_once_with(store_uri=mlflow.get_tracking_uri())
        request.assert_called_once()
        kwargs = request.call_args.kwargs
        suffix = "endpoints/ae-weather" if endpoint else "aliases/prod"
        assert request.call_args.args == (
            "GET",
            f"{URI}/api/3.0/mlflow/mcp-servers/com.example%2Fweather/{suffix}",
        )
        assert kwargs["headers"].get(WORKSPACE_HEADER_NAME) == (
            expected_workspace if supports_workspaces else None
        )


@pytest.mark.parametrize("server", [_version(workspace="other"), _endpoint(workspace="other")])
def test_workspace_mismatch_before_lookup(server):
    with mock.patch("mlflow.tracing.utils.mcp._get_store") as client:
        with pytest.raises(MlflowException, match="does not match trace workspace"):
            _resolve(server)
        client.assert_not_called()


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("trace_workspace", [None, "trace-workspace"])
def test_span_rejects_entity_from_current_workspace_when_trace_workspace_differs(
    endpoint, trace_workspace
):
    server = _endpoint(workspace="different") if endpoint else _version(workspace="different")
    with mock.patch("mlflow.tracing.utils.mcp._get_store") as get_store:
        with ServerWorkspaceContext(trace_workspace):
            with mlflow.start_span("mcp") as span:
                before = span.attributes.copy()
                with ServerWorkspaceContext("different"):
                    with pytest.raises(MlflowException, match="does not match trace workspace"):
                        span.link_mcp_server(server, tool="forecast")
                assert span.attributes == before
        get_store.assert_not_called()


def test_child_span_does_not_inherit_version_or_tool():
    with mlflow.start_span("parent") as parent:
        parent.link_mcp_server(_version(), tool="forecast")
        with mlflow.start_span("discovery") as child:
            child.link_mcp_server(MCPServerRef(NAME))
            assert child.get_attribute(SpanAttributeKey.MCP_SERVER_NAME) == NAME
            assert SpanAttributeKey.MCP_SERVER_VERSION not in child.attributes
            assert SpanAttributeKey.MCP_TOOL_NAME not in child.attributes


@pytest.mark.parametrize("trace_workspace", [None, "trace-workspace"])
def test_resolution_uses_captured_trace_workspace(trace_workspace, monkeypatch, registry_store):
    expected = trace_workspace or "default"

    def lookup(name, alias):
        assert get_request_workspace() == expected
        return _version(workspace=expected)

    with (
        mock.patch.object(
            registry_store, "get_mcp_server_version_by_alias", side_effect=lookup
        ) as lookup_mock,
        monkeypatch.context() as workspace_env,
    ):
        with ServerWorkspaceContext(trace_workspace):
            with mlflow.start_span("mcp") as span:
                with ServerWorkspaceContext("different"):
                    workspace_env.setenv("MLFLOW_WORKSPACE", "different")
                    span.link_mcp_server(MCPServerRef(NAME, alias="prod"))
                    assert get_request_workspace() == "different"
        lookup_mock.assert_called_once_with(name=NAME, alias="prod")


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("workspaces", [(None, "team-a"), ("team-a", "team-b")])
def test_concurrent_resolution_isolates_and_restores_workspaces(
    endpoint, fail_first, workspaces, monkeypatch, registry_store
):
    barrier = Barrier(2, timeout=5)
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
    failed_workspace = workspaces[0] or "default"

    def lookup(**kwargs):
        workspace = get_request_workspace()
        barrier.wait()
        assert get_request_workspace() == workspace
        if fail_first and workspace == failed_workspace:
            raise MlflowException("lookup failed")
        version = _version(version=workspace, workspace=workspace)
        return _endpoint(workspace=workspace, resolved_version=version) if endpoint else version

    def resolve(workspace):
        expected = workspace or "default"
        server = _endpoint(workspace=expected) if endpoint else MCPServerRef(NAME, alias="prod")
        with ServerWorkspaceContext(workspace):
            if fail_first and expected == failed_workspace:
                with pytest.raises(MlflowException, match="lookup failed"):
                    _resolve(server, workspace=expected)
            else:
                assert _resolve(server, workspace=expected).version == expected
            assert get_request_workspace() == workspace
        assert get_request_workspace() == "environment-workspace"

    with (
        mock.patch.object(registry_store, method, side_effect=lookup) as lookup_mock,
        monkeypatch.context() as workspace_env,
        ServerWorkspaceContext("main-workspace"),
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-mcp-workspace") as executor,
    ):
        workspace_env.setenv("MLFLOW_WORKSPACE", "environment-workspace")
        list(executor.map(resolve, workspaces))
        assert lookup_mock.call_count == 2
        assert get_request_workspace() == "main-workspace"
        assert os.environ["MLFLOW_WORKSPACE"] == "environment-workspace"
    assert len(_resolution_cache) == (1 if fail_first else 2)


@pytest.mark.parametrize(
    ("method", "server", "result"),
    [
        (
            "get_mcp_server_version_by_alias",
            MCPServerRef(NAME, alias="prod"),
            _version(workspace="other"),
        ),
        (
            "get_mcp_server_version_by_alias",
            MCPServerRef(NAME, alias="prod"),
            _version(name="com.example/other"),
        ),
        ("get_mcp_access_endpoint", _endpoint(), _endpoint(workspace="other")),
        (
            "get_mcp_access_endpoint",
            _endpoint(),
            _endpoint(resolved_version=_version(workspace="other")),
        ),
        ("get_mcp_access_endpoint", _endpoint(), _endpoint(id="ae-other")),
    ],
)
def test_invalid_resolution_not_cached(method, server, result, registry_store):
    with mock.patch.object(registry_store, method, return_value=result) as lookup:
        with pytest.raises(MlflowException, match="does not match"):
            _resolve(server)
        assert len(_resolution_cache) == 0
        lookup.assert_called_once()


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize(
    ("env_ttl", "override", "expiry"),
    [(None, None, 60), ("12", None, 12), ("12", 3, 3), ("0", 10, 10)],
)
def test_cache_ttl_and_expiry(endpoint, env_ttl, override, expiry, monkeypatch, registry_store):
    if env_ttl is not None:
        monkeypatch.setenv("MLFLOW_MCP_SERVER_CACHE_TTL_SECONDS", env_ttl)
    now = 100.0
    server = _endpoint() if endpoint else MCPServerRef(NAME, alias="prod")
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
    versions = [_version(), _version("2.0.0")]
    results = (
        [_endpoint(resolved_version=version) for version in versions] if endpoint else versions
    )
    with (
        mock.patch("mlflow.tracing.utils.mcp.monotonic", side_effect=lambda: now) as clock,
        mock.patch.object(registry_store, method, side_effect=results) as lookup,
    ):
        first = _resolve(server, cache_ttl_seconds=override)
        assert first.version == "1.0.0"
        now += expiry - 0.1
        assert _resolve(server, cache_ttl_seconds=override).version == "1.0.0"
        lookup.assert_called_once()
        now += 0.1
        assert _resolve(server, cache_ttl_seconds=override).version == "2.0.0"
        assert first.version == "1.0.0"
        assert lookup.call_count == 2
        clock.assert_called()


@pytest.mark.parametrize("endpoint", [False, True])
def test_cache_freshness_uses_each_call_ttl(endpoint, registry_store):
    now = 100.0
    server = _endpoint() if endpoint else MCPServerRef(NAME, alias="prod")
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
    versions = [_version(), _version("2.0.0")]
    results = (
        [_endpoint(resolved_version=version) for version in versions] if endpoint else versions
    )
    with (
        mock.patch("mlflow.tracing.utils.mcp.monotonic", side_effect=lambda: now) as clock,
        mock.patch.object(registry_store, method, side_effect=results) as lookup,
    ):
        assert _resolve(server, cache_ttl_seconds=60).version == "1.0.0"
        now += 10
        assert _resolve(server, cache_ttl_seconds=60).version == "1.0.0"
        assert _resolve(server, cache_ttl_seconds=5).version == "2.0.0"
        now += 1
        assert _resolve(server, cache_ttl_seconds=60).version == "2.0.0"
        assert lookup.call_count == 2
        clock.assert_called()


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("first_finishes_first", [False, True])
def test_late_lookup_does_not_replace_newer_cached_resolution(
    endpoint, first_finishes_first, registry_store
):
    first_started = Event()
    second_started = Event()
    release_first = Event()
    release_second = Event()
    now = 100.0
    server = _endpoint() if endpoint else MCPServerRef(NAME, alias="prod")
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"

    def lookup(**kwargs):
        first = not first_started.is_set()
        version = _version("1.0.0" if first else "2.0.0")
        (first_started if first else second_started).set()
        assert (release_first if first else release_second).wait(timeout=5)
        return _endpoint(resolved_version=version) if endpoint else version

    with (
        mock.patch("mlflow.tracing.utils.mcp.monotonic", side_effect=lambda: now) as clock,
        mock.patch.object(registry_store, method, side_effect=lookup) as lookup_mock,
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-mcp-cache") as executor,
    ):
        first = executor.submit(_resolve, server)
        try:
            assert first_started.wait(timeout=5)
            now += 1
            second = executor.submit(_resolve, server)
            assert second_started.wait(timeout=5)
            ordered = (
                [(release_first, first), (release_second, second)]
                if first_finishes_first
                else [(release_second, second), (release_first, first)]
            )
            for release, future in ordered:
                now += 1
                release.set()
                future.result(timeout=5)
        finally:
            release_first.set()
            release_second.set()
        assert first.result(timeout=5).version == "1.0.0"
        assert second.result(timeout=5).version == "2.0.0"
        assert _resolve(server).version == "2.0.0"
        assert lookup_mock.call_count == 2
        clock.assert_called()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
@pytest.mark.parametrize("endpoint", [False, True])
def test_fork_resets_cache_and_inherited_lock(endpoint, registry_store):
    server = _endpoint() if endpoint else MCPServerRef(NAME, alias="prod")
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
    versions = [_version("1"), _version("2")]
    results = (
        [_endpoint(resolved_version=version) for version in versions] if endpoint else versions
    )
    with mock.patch.object(registry_store, method, side_effect=results) as lookup:
        assert _resolve(server).version == "1"
        read_fd, write_fd = os.pipe()
        child_reaped = False
        old_lock = mcp._cache_lock
        with old_lock:
            try:
                pid = os.fork()
                if pid == 0:
                    os.close(read_fd)
                    try:
                        os.write(write_fd, _resolve(server).version.encode())
                        os._exit(0)
                    except BaseException:
                        os._exit(1)

                os.close(write_fd)
                ready, _, _ = select.select([read_fd], [], [], 5)
                assert ready
                assert os.read(read_fd, 1) == b"2"
                _, status = os.waitpid(pid, 0)
                child_reaped = True
                assert os.waitstatus_to_exitcode(status) == 0
            finally:
                os.close(read_fd)
                if "pid" in locals() and pid != 0 and not child_reaped:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
        assert _resolve(server).version == "1"
        lookup.assert_called_once()


@pytest.mark.parametrize("env_ttl", ["0", "60"])
@pytest.mark.parametrize("endpoint", [False, True])
def test_zero_bypasses_existing_cache(env_ttl, endpoint, monkeypatch, registry_store):
    monkeypatch.setenv("MLFLOW_MCP_SERVER_CACHE_TTL_SECONDS", env_ttl)
    server = _endpoint() if endpoint else MCPServerRef(NAME, alias="prod")
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
    versions = [_version(), _version("2.0.0"), _version("3.0.0")]
    results = (
        [_endpoint(resolved_version=version) for version in versions] if endpoint else versions
    )
    with mock.patch.object(
        registry_store,
        method,
        side_effect=results,
    ) as lookup:
        assert _resolve(server, cache_ttl_seconds=60).version == "1.0.0"
        ttl = None if env_ttl == "0" else 0
        assert _resolve(server, cache_ttl_seconds=ttl).version == "2.0.0"
        assert _resolve(server, cache_ttl_seconds=ttl).version == "3.0.0"
        assert lookup.call_count == 3


@pytest.mark.parametrize("failure", ["alias_missing", "endpoint_missing", "url", "transport"])
def test_failed_cache_refresh_preserves_span_and_historical_identity(failure, registry_store):
    is_endpoint = failure != "alias_missing"
    server = _endpoint() if is_endpoint else MCPServerRef(NAME, alias="prod")
    method = "get_mcp_access_endpoint" if is_endpoint else "get_mcp_server_version_by_alias"
    initial = _endpoint(resolved_version=_version()) if is_endpoint else _version()
    match failure:
        case "url":
            refreshed = replace(initial, url="https://example.com/changed")
            error = "URL or transport has changed"
        case "transport":
            refreshed = replace(initial, transport_type=MCPRemoteTransportType.SSE)
            error = "URL or transport has changed"
        case _:
            refreshed = MlflowException("not found")
            error = "not found"
    now = 100.0
    with (
        mock.patch("mlflow.tracing.utils.mcp.monotonic", side_effect=lambda: now) as clock,
        mock.patch.object(registry_store, method, side_effect=[initial, refreshed]) as lookup,
    ):
        with mlflow.start_span("first") as first:
            first.link_mcp_server(server, tool="forecast")
        with mlflow.start_span("second") as second:
            second.link_mcp_server(server, tool="forecast")
            lookup.assert_called_once()
            before = second.attributes.copy()
            now += 60
            with pytest.raises(MlflowException, match=error):
                second.link_mcp_server(server, tool="different")
            assert second.attributes == before
        assert first.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == "1.0.0"
        assert lookup.call_count == 2
        clock.assert_called()


@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf"), "60"])
def test_invalid_ttl(ttl):
    with pytest.raises(MlflowException, match="cache TTL"):
        _resolve(MCPServerRef(NAME), cache_ttl_seconds=ttl)


@pytest.mark.parametrize("endpoint", [False, True])
def test_cache_isolated_by_tracking_uri_workspace_and_lookup_key(endpoint, registry_store):
    method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"

    def resolve(**kwargs):
        version = _version(workspace=get_request_workspace())
        return (
            _endpoint(
                id=kwargs["endpoint_id"],
                workspace=get_request_workspace(),
                resolved_version=version,
            )
            if endpoint
            else version
        )

    with mock.patch.object(
        registry_store,
        method,
        side_effect=resolve,
    ) as lookup:
        for uri in [URI, "http://other.example.com"]:
            for workspace in ["default", "other"]:
                for alias in ["prod", "staging"]:
                    server = (
                        _endpoint(id=alias, workspace=workspace)
                        if endpoint
                        else MCPServerRef(NAME, alias=alias)
                    )
                    for _ in range(2):
                        _resolve(server, tracking_uri=uri, workspace=workspace)
        assert lookup.call_count == 8


@skip_when_testing_trace_sdk
@pytest.mark.parametrize(
    "version",
    [None, "1", "true", "null", "[]", "{}", '"quoted"', "1.0.0+Build", " 1.0.0", "1.0.0 "],
)
@pytest.mark.parametrize("annotation", ["link", "mlflow", "otel"])
def test_exported_attributes_create_backend_link(async_logging_enabled, version, annotation):
    with mlflow.start_span("mcp", span_type="TOOL") as span:
        if annotation == "link":
            span.link_mcp_server(MCPServerRef(NAME, version=version), tool="forecast")
        else:
            target = span._span if annotation == "otel" else span
            target.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, NAME)
            target.set_attribute(SpanAttributeKey.MCP_TOOL_NAME, "forecast")
            if version is not None:
                target.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, version)
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(span.trace_id)
    assert trace.data.spans[0].get_attribute(SpanAttributeKey.MCP_SERVER_NAME) == NAME
    assert trace.data.spans[0].get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == version
    assert trace.data.spans[0].get_attribute(SpanAttributeKey.MCP_TOOL_NAME) == "forecast"
    assert mlflow.MlflowClient()._tracking_client.store.get_mcp_servers_for_trace(
        span.trace_id
    ) == [TraceMCPServerLink("default", NAME, version)]


@pytest.mark.parametrize("version", [None, "1", "true", "null"])
@pytest.mark.parametrize("workspace", [None, "team-a"])
@pytest.mark.parametrize("native_otel", [False, True])
def test_artifact_export_fallback_preserves_mcp_attributes(
    version, workspace, native_otel, async_logging_enabled, monkeypatch
):
    with ServerWorkspaceContext(workspace):
        span = mlflow.start_span_no_context("mcp")
        if native_otel:
            span._span.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, NAME)
            span._span.set_attribute(SpanAttributeKey.MCP_TOOL_NAME, "forecast")
            if version is not None:
                span._span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, version)
        else:
            span.link_mcp_server(MCPServerRef(NAME, version=version), tool="forecast")
    exporter = _get_trace_exporter()
    observed_workspaces = []

    def start_trace(trace_info):
        observed_workspaces.append(get_request_workspace())
        trace_info.tags.pop(TraceTagKey.SPANS_LOCATION, None)
        return trace_info

    def upload_trace_data(trace_json):
        observed_workspaces.append(get_request_workspace())

    artifact_repo = mock.MagicMock(upload_trace_data=mock.Mock(side_effect=upload_trace_data))
    with (
        mock.patch.object(exporter._client, "log_spans", side_effect=NotImplementedError) as log,
        mock.patch.object(exporter._client, "start_trace", side_effect=start_trace) as start,
        mock.patch.object(
            exporter._client, "_get_artifact_repo_for_trace", return_value=artifact_repo
        ) as get_repo,
        ServerWorkspaceContext("different"),
        monkeypatch.context() as workspace_env,
    ):
        workspace_env.setenv("MLFLOW_WORKSPACE", "different")
        span.end()
        mlflow.flush_trace_async_logging()
        log.assert_called_once()
        start.assert_called_once()
        get_repo.assert_called_once_with(start.call_args.args[0])
        artifact_repo.upload_trace_data.assert_called_once()
        assert get_request_workspace() == "different"
    assert not exporter._store_supports_log_spans
    assert observed_workspaces == [workspace, workspace]
    uploaded = TraceData.from_dict(
        json.loads(artifact_repo.upload_trace_data.call_args.args[0])
    ).spans[0]
    expected = {SpanAttributeKey.MCP_SERVER_NAME: NAME, SpanAttributeKey.MCP_TOOL_NAME: "forecast"}
    if version is not None:
        expected[SpanAttributeKey.MCP_SERVER_VERSION] = version
    assert {
        key: value for key, value in uploaded.attributes.items() if key.startswith("mlflow.mcp.")
    } == expected


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("child", [False, True])
def test_resolution_stays_with_trace_tracking_server(
    endpoint, child, async_logging_enabled, monkeypatch
):
    tracking_uri = mlflow.get_tracking_uri()
    with monkeypatch.context() as mp:
        mp.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", None)
        mp.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        store = client._tracking_client.store
        store.create_mcp_server_version({"name": NAME, "version": "1.0.0"}, tools=[])
        store.set_mcp_server_alias(NAME, "prod", "1.0.0")
        server = (
            store.create_mcp_access_endpoint(NAME, "https://example.com/mcp", server_alias="prod")
            if endpoint
            else MCPServerRef(NAME, alias="prod")
        )
        method = "get_mcp_access_endpoint" if endpoint else "get_mcp_server_version_by_alias"
        response = (
            replace(server, resolved_version=_version("2.0.0")) if endpoint else _version("2.0.0")
        )
        other_store = mock.MagicMock(
            supports_workspaces=False, **{f"{method}.return_value": response}
        )

        with mlflow.start_span("mcp") as span:
            mp.setenv("MLFLOW_TRACKING_URI", URI)
            with mock.patch(
                "mlflow.tracing.utils.mcp._get_store",
                side_effect=lambda store_uri: store if store_uri == tracking_uri else other_store,
            ) as get_store:
                with mlflow.start_span("child") if child else nullcontext(span) as linked_span:
                    linked_span.link_mcp_server(server)
                    assert linked_span.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == "1.0.0"
                get_store.assert_called_once_with(store_uri=tracking_uri)
                getattr(other_store, method).assert_not_called()
        mlflow.flush_trace_async_logging()
        assert store.get_mcp_servers_for_trace(span.trace_id) == [
            TraceMCPServerLink("default", NAME, "1.0.0")
        ]


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("workspace", [None, "default"])
@pytest.mark.parametrize("batch", [False, True])
def test_link_export_preserves_captured_default_workspace(
    workspace, batch, async_logging_enabled, monkeypatch
):
    monkeypatch.setenv("MLFLOW_USE_BATCH_SPAN_PROCESSOR", str(batch))
    with ServerWorkspaceContext(workspace):
        span = mlflow.start_span_no_context("mcp")
        span.link_mcp_server(MCPServerRef(NAME), tool="forecast")
    exporter = _get_trace_exporter()
    log_spans = exporter._client.log_spans
    start_trace = exporter._client.start_trace
    observed_workspaces = []

    def capture_log_spans(*args, **kwargs):
        observed_workspaces.append(get_request_workspace())
        return log_spans(*args, **kwargs)

    def capture_start_trace(*args, **kwargs):
        observed_workspaces.append(get_request_workspace())
        return start_trace(*args, **kwargs)

    with (
        mock.patch.object(exporter._client, "log_spans", side_effect=capture_log_spans) as log,
        mock.patch.object(
            exporter._client, "start_trace", side_effect=capture_start_trace
        ) as start,
        ServerWorkspaceContext("different"),
    ):
        monkeypatch.setenv("MLFLOW_WORKSPACE", "different")
        span.end()
        mlflow.flush_trace_async_logging()
        assert get_request_workspace() == "different"
        log.assert_called_once()
        start.assert_called_once()
    assert observed_workspaces == [workspace, workspace]
    assert exporter._client.store.get_mcp_servers_for_trace(span.trace_id) == [
        TraceMCPServerLink("default", NAME, None)
    ]


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("workspace", ["default", "team-a"])
def test_registry_resolution_and_historical_links(workspace, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    client = mlflow.MlflowClient()
    if workspace != "default":
        client.create_workspace(workspace)
    with WorkspaceContext(workspace):
        mlflow.set_experiment("mcp-trace-linking")
        store = client._tracking_client.store
        for version in ["1.0.0", "2.0.0"]:
            store.create_mcp_server_version({"name": NAME, "version": version}, tools=[])
        client.set_mcp_server_alias(NAME, "prod", "1.0.0")
        endpoint = client.create_mcp_access_endpoint(
            NAME, "https://example.com/mcp", server_alias="prod"
        )
        with mlflow.start_span("first") as first:
            with mlflow.start_span("alias") as alias_span:
                alias_span.link_mcp_server(MCPServerRef(NAME, alias="prod"), cache_ttl_seconds=0)
            with mlflow.start_span("endpoint") as endpoint_span:
                endpoint_span.link_mcp_server(endpoint, cache_ttl_seconds=0)
        mlflow.flush_trace_async_logging()
        client.set_mcp_server_alias(NAME, "prod", "2.0.0")
        with mlflow.start_span("second") as second:
            second.link_mcp_server(endpoint, cache_ttl_seconds=0)
        mlflow.flush_trace_async_logging()
        assert store.get_mcp_servers_for_trace(first.trace_id) == [
            TraceMCPServerLink(workspace, NAME, "1.0.0")
        ]
        assert store.get_mcp_servers_for_trace(second.trace_id) == [
            TraceMCPServerLink(workspace, NAME, "2.0.0")
        ]
        assert (
            mlflow
            .get_trace(first.trace_id)
            .data.spans[1]
            .get_attribute(SpanAttributeKey.MCP_SERVER_VERSION)
            == "1.0.0"
        )
        with mlflow.start_span("invalid") as invalid:
            before = invalid.attributes.copy()
            with pytest.raises(MlflowException, match="Alias 'missing' not found"):
                invalid.link_mcp_server(MCPServerRef(NAME, alias="missing"), cache_ttl_seconds=0)
            assert invalid.attributes == before
            client.delete_mcp_access_endpoint(NAME, endpoint.id)
            with pytest.raises(MlflowException, match="not found"):
                invalid.link_mcp_server(endpoint, cache_ttl_seconds=0)
            assert invalid.attributes == before
