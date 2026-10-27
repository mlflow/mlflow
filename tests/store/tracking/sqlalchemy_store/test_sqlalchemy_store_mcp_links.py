from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite

from mlflow.entities.trace_mcp_server_link import MCPServerSpanObservation, TraceMCPServerLink
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlSpan, SqlSpanMCPServerLink, SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import _get_filter_clauses_for_search_traces
from mlflow.tracing.constant import SpanAttributeKey, SpansLocation, TraceTagKey
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.store.tracking.sqlalchemy_store.conftest import _create_trace, create_test_span

pytestmark = pytest.mark.notrackingurimock


@pytest.mark.parametrize("case_insensitive", [False, True])
def test_mcp_link_summaries_preserve_exact_versions(store_and_trace_info, case_insensitive):
    store, trace = store_and_trace_info
    if case_insensitive:
        if store.db_type != "sqlite":
            pytest.skip("Uses SQLite NOCASE to exercise database-side string equivalence")
        with store.engine.begin() as conn:
            table = sa.Table("span_mcp_server_links", sa.MetaData(), autoload_with=conn)
            table.c.mcp_version.type = sa.String(256, collation="NOCASE")
            table.drop(conn)
            table.create(conn)
    versions = ["1.0.0+Build", "1.0.0+build", "1.0.0+build "]
    store.upsert_span_mcp_server_links(
        trace_id=trace.trace_id,
        links=[
            MCPServerSpanObservation(str(index), "com.example/weather", version)
            for index, version in enumerate([*versions, versions[0]])
        ],
    )
    summaries = store.get_mcp_servers_for_trace(trace_id=trace.trace_id)
    assert len(summaries) == len(versions)
    assert {summary.version for summary in summaries} == set(versions)


def test_mcp_links_summarize_observations_without_spans_or_registry(store_and_trace_info):
    store, trace = store_and_trace_info
    assert store.get_mcp_servers_for_trace(trace.trace_id) == []
    links = [
        MCPServerSpanObservation("span-1", "com.example/weather", "1.0.0"),
        MCPServerSpanObservation("span-2", "com.example/weather", "1.0.0"),
        MCPServerSpanObservation("span-3", "com.example/weather"),
        MCPServerSpanObservation("span-4", "com.example/weather", "2.0.0"),
        MCPServerSpanObservation("span-1", "com.example/other"),
    ]
    store.upsert_span_mcp_server_links(trace.trace_id, links)
    store.upsert_span_mcp_server_links(trace.trace_id, links)

    assert set(store.get_mcp_servers_for_trace(trace.trace_id)) == {
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", "1.0.0"),
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", "2.0.0"),
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", None),
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/other", None),
    }
    with store.ManagedSessionMaker() as session:
        assert (
            session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace.trace_id).count()
        ) == len(links)


@pytest.mark.parametrize("version", ["2.0.0", None])
def test_mcp_links_update_version_preserving_creation_time(store_and_trace_info, version):
    store, trace = store_and_trace_info
    store.upsert_span_mcp_server_links(
        trace.trace_id, [MCPServerSpanObservation("span-1", "com.example/weather", "1.0.0")]
    )
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace.trace_id).update({
            "created_at": 123
        })

    store.upsert_span_mcp_server_links(
        trace.trace_id, [MCPServerSpanObservation("span-1", "com.example/weather", version)]
    )
    with store.ManagedSessionMaker() as session:
        row = session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace.trace_id).one()
        assert row.mcp_version == version
        assert row.created_at == 123


def test_mcp_links_duplicate_batch_keys_use_last_observation(store_and_trace_info):
    store, trace = store_and_trace_info
    store.upsert_span_mcp_server_links(
        trace.trace_id,
        [
            MCPServerSpanObservation("span-1", "com.example/weather", "1.0.0"),
            MCPServerSpanObservation("span-1", "com.example/weather", "2.0.0"),
        ],
    )
    assert store.get_mcp_servers_for_trace(trace.trace_id) == [
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", "2.0.0")
    ]


@pytest.mark.parametrize(
    ("initial_version", "updated_version"),
    [(None, "1.0.0"), ("1.0.0", "2.0.0"), ("1.0.0", None)],
)
def test_log_spans_updates_mcp_version(store, initial_version, updated_version):
    experiment_id = store.create_experiment("mcp-version-update")
    for version in [initial_version, updated_version]:
        attributes = {SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"}
        if version is not None:
            attributes[SpanAttributeKey.MCP_SERVER_VERSION] = version
        span = create_test_span("trace-mcp-version-update", attributes=attributes)
        store.log_spans(experiment_id, [span])
        assert store.get_mcp_servers_for_trace(span.trace_id) == [
            TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", version)
        ]
        for query_version in ["1.0.0", "2.0.0"]:
            traces, _ = store.search_traces(
                [experiment_id],
                filter_string=(
                    f"mcp_server = 'com.example/weather' AND mcp_server.version = '{query_version}'"
                ),
            )
            assert [trace.trace_id for trace in traces] == (
                [span.trace_id] if version == query_version else []
            )
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlSpanMCPServerLink).filter_by(trace_id=span.trace_id).count() == 1
    assert (
        store
        .get_trace(span.trace_id)
        .data.spans[0]
        .get_attribute(SpanAttributeKey.MCP_SERVER_VERSION)
        == updated_version
    )


def test_mcp_links_survive_archival_and_cascade_on_delete(store, tmp_path):
    experiment_id = store.create_experiment("mcp-archival")
    trace_id = "trace-mcp-archival"
    store.log_spans(
        experiment_id,
        [
            create_test_span(
                trace_id,
                span_id=index,
                attributes={
                    SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                    **({SpanAttributeKey.MCP_SERVER_VERSION: version} if version else {}),
                },
            )
            for index, version in enumerate([None, "1.0.0"], start=1)
        ],
    )
    assert (
        store.archive_traces(
            resolved_trace_archival_location=(tmp_path / "archive").as_uri(),
            broader_retention="1d",
        )
        == 1
    )
    info = store.get_trace_info(trace_id)
    assert info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
    with store.ManagedSessionMaker() as session:
        assert {row.content for row in session.query(SqlSpan).filter_by(trace_id=trace_id)} == {""}
    assert set(store.get_mcp_servers_for_trace(trace_id)) == {
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", version)
        for version in [None, "1.0.0"]
    }
    for filter_string in [
        "mcp_server = 'com.example/weather'",
        "mcp_server = 'com.example/weather' AND mcp_server.version = '1.0.0'",
    ]:
        traces, _ = store.search_traces([experiment_id], filter_string=filter_string)
        assert [trace.trace_id for trace in traces] == [trace_id]
    assert len(store.get_trace(trace_id).data.spans) == 2
    assert store.delete_traces(experiment_id, trace_ids=[trace_id]) == 1
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace_id).count() == 0
    traces, _ = store.search_traces(
        [experiment_id], filter_string="mcp_server = 'com.example/weather'"
    )
    assert traces == []


def test_mcp_links_batch_is_atomic(store_and_trace_info):
    store, trace = store_and_trace_info
    links = [MCPServerSpanObservation(f"span-{i}", "com.example/weather") for i in range(100)]
    links.append(MCPServerSpanObservation("span-invalid", None))
    with pytest.raises(MlflowException, match="mcp_name"):
        store.upsert_span_mcp_server_links(trace.trace_id, links)
    assert store.get_mcp_servers_for_trace(trace.trace_id) == []


def test_mcp_links_empty_batch(store_and_trace_info):
    store, trace = store_and_trace_info
    store.upsert_span_mcp_server_links(trace.trace_id, [])
    assert store.get_mcp_servers_for_trace(trace.trace_id) == []


@pytest.mark.parametrize("operation", ["get", "upsert", "empty-upsert"])
def test_mcp_links_require_existing_trace(store, operation):
    if operation == "get":
        method = store.get_mcp_servers_for_trace
        args = ("missing",)
    else:
        method = store.upsert_span_mcp_server_links
        links = [] if operation == "empty-upsert" else [MCPServerSpanObservation("s", "n")]
        args = ("missing", links)
    with pytest.raises(MlflowException, match="Trace with ID 'missing' not found") as exc:
        method(*args)
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_log_spans_materializes_mcp_links(store, caplog):
    experiment_id = store.create_experiment("mcp-ingestion")
    spans = [
        create_test_span(
            "trace-mcp",
            span_id=1,
            attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"},
        ),
        create_test_span(
            "trace-mcp",
            span_id=2,
            attributes={
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0+Build.1",
                "mlflow.mcp.tool.name": "forecast",
            },
        ),
        create_test_span(
            "trace-other",
            span_id=3,
            attributes={
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/other",
                SpanAttributeKey.MCP_SERVER_VERSION: "latest",
            },
        ),
        create_test_span("trace-plain", span_id=4),
    ]
    store.log_spans(experiment_id, spans)
    store.log_spans(experiment_id, spans)
    assert set(store.get_mcp_servers_for_trace("trace-mcp")) == {
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", None),
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/weather", "1.0.0+Build.1"),
    }
    assert store.get_mcp_servers_for_trace("trace-other") == [
        TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, "com.example/other", "latest")
    ]
    assert store.get_mcp_servers_for_trace("trace-plain") == []
    stored_spans = store.get_trace("trace-mcp").data.spans
    assert (
        next(s for s in stored_spans if s.span_id == spans[1].span_id).get_attribute(
            "mlflow.mcp.tool.name"
        )
        == "forecast"
    )
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlSpanMCPServerLink).count() == 3
    assert "malformed MCP" not in caplog.text


@pytest.mark.parametrize(
    ("attributes", "expected_version", "expected_name", "warning"),
    [
        ({SpanAttributeKey.MCP_SERVER_NAME: ""}, None, None, "name"),
        ({SpanAttributeKey.MCP_SERVER_NAME: 123}, None, None, "name"),
        ({SpanAttributeKey.MCP_SERVER_NAME: ["com.example/weather"]}, None, None, "name"),
        ({SpanAttributeKey.MCP_SERVER_NAME: "https://example.com/mcp"}, None, None, "name"),
        ({SpanAttributeKey.MCP_SERVER_NAME: "com.example/" + "x" * 256}, None, None, "name"),
        ({SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0"}, None, None, "name"),
        (
            {
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                SpanAttributeKey.MCP_SERVER_VERSION: 123,
            },
            None,
            "com.example/weather",
            "version",
        ),
        (
            {
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                SpanAttributeKey.MCP_SERVER_VERSION: "",
            },
            None,
            "com.example/weather",
            "version",
        ),
        (
            {
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                SpanAttributeKey.MCP_SERVER_VERSION: " \t\n",
            },
            None,
            "com.example/weather",
            "version",
        ),
        (
            {
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                SpanAttributeKey.MCP_SERVER_VERSION: "x" * 257,
            },
            None,
            "com.example/weather",
            "version",
        ),
        (
            {
                SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0\x00invalid",
            },
            None,
            "com.example/weather",
            "version",
        ),
        *[
            (
                {
                    SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                    SpanAttributeKey.MCP_SERVER_VERSION: version,
                },
                None,
                "com.example/weather",
                "version",
            )
            for version in ["1.0.0\ud800", "1.0.0\udfff"]
        ],
    ],
)
def test_log_spans_ignores_malformed_mcp_attributes(
    store, caplog, attributes, expected_version, expected_name, warning
):
    experiment_id = store.create_experiment("malformed-mcp-ingestion")
    span = create_test_span("trace-malformed-mcp", attributes=attributes)
    other_span = create_test_span("trace-unrelated", span_id=222)
    store.log_spans(experiment_id, [span, other_span])
    assert len(store.get_trace(span.trace_id).data.spans) == 1
    assert len(store.get_trace(other_span.trace_id).data.spans) == 1
    expected = (
        [TraceMCPServerLink(DEFAULT_WORKSPACE_NAME, expected_name, expected_version)]
        if expected_name
        else []
    )
    assert store.get_mcp_servers_for_trace(span.trace_id) == expected
    assert f"malformed MCP server {warning}" in caplog.text


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("com.example/" + "x" * (256 - len("com.example/")), "1.0.0"),
        ("com.example/weather", "x" * 256),
    ],
)
def test_log_spans_accepts_mcp_attributes_at_length_limit(store, caplog, name, version):
    experiment_id = store.create_experiment("mcp-attribute-length-limit")
    span = create_test_span(
        "trace-mcp-length-limit",
        attributes={
            SpanAttributeKey.MCP_SERVER_NAME: name,
            SpanAttributeKey.MCP_SERVER_VERSION: version,
        },
    )
    store.log_spans(experiment_id, [span])
    assert store.get_mcp_servers_for_trace(span.trace_id) == [
        TraceMCPServerLink(workspace=DEFAULT_WORKSPACE_NAME, name=name, version=version)
    ]
    assert len(store.get_trace(span.trace_id).data.spans) == 1
    assert "malformed MCP" not in caplog.text


@pytest.mark.parametrize("existing_trace", [False, True])
def test_log_spans_rolls_back_mcp_links_and_spans(store, existing_trace):
    experiment_id = store.create_experiment("atomic-mcp-ingestion")
    trace_id = "trace-atomic-mcp"
    if existing_trace:
        _create_trace(store, trace_id, experiment_id=experiment_id)
    span = create_test_span(
        trace_id, attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"}
    )
    upsert = store._upsert_span_mcp_server_links

    def upsert_then_fail(*args):
        upsert(*args)
        raise RuntimeError("failure after MCP link write")

    with mock.patch.object(
        store, "_upsert_span_mcp_server_links", side_effect=upsert_then_fail
    ) as m:
        with pytest.raises(MlflowException, match="failure after MCP link write"):
            store.log_spans(experiment_id, [span])
        m.assert_called_once()
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlSpan).filter_by(trace_id=trace_id).count() == 0
        assert session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace_id).count() == 0
    if existing_trace:
        assert store.get_trace_info(trace_id).trace_id == trace_id
    else:
        with pytest.raises(MlflowException, match="not found"):
            store.get_trace_info(trace_id)


@pytest.fixture
def store_with_mcp_links(store):
    experiment_id = store.create_experiment("mcp-search")
    for trace_id, timestamp, environment in [
        ("tr-a", 30, "prod"),
        ("tr-b", 20, "prod"),
        ("tr-c", 10, "dev"),
        ("tr-d", 40, "prod"),
    ]:
        _create_trace(
            store,
            trace_id,
            experiment_id=experiment_id,
            request_time=timestamp,
            tags={"env": environment},
        )
    for trace_id, links in {
        "tr-a": [
            MCPServerSpanObservation("s1", "com.example/weather", "1.0.0"),
            MCPServerSpanObservation("s2", "com.example/weather", "1.0.0"),
            MCPServerSpanObservation("s3", "com.example/search", "2.0.0"),
        ],
        "tr-b": [MCPServerSpanObservation("s1", "com.example/weather")],
        "tr-c": [MCPServerSpanObservation("s1", "com.example/weather", "2.0.0")],
    }.items():
        store.upsert_span_mcp_server_links(trace_id, links)
    return store, experiment_id


@pytest.mark.parametrize(
    ("filter_string", "expected_ids"),
    [
        ("mcp_server = 'com.example/weather'", ["tr-a", "tr-b", "tr-c"]),
        ("mcp_server = 'com.example/weather' AND mcp_server.version = '1.0.0'", ["tr-a"]),
        ("mcp_server = 'com.example/weather' AND mcp_server.version = '2.0.0'", ["tr-c"]),
        ("mcp_server.version = '2.0.0' AND mcp_server = 'com.example/weather'", ["tr-c"]),
        ("mcp_server = 'com.example/search' AND mcp_server.version = '1.0.0'", []),
        ("mcp_server.version = '2.0.0'", ["tr-a", "tr-c"]),
        ("mcp_server = 'com.example/missing'", []),
        ("mcp_server = 'com.example/Weather'", []),
        ("mcp_server = 'com.example/weather' AND timestamp >= 20", ["tr-a", "tr-b"]),
        ("mcp_server = 'com.example/weather' AND tag.env = 'prod'", ["tr-a", "tr-b"]),
    ],
)
def test_search_traces_mcp_filters(store_with_mcp_links, filter_string, expected_ids):
    store, experiment_id = store_with_mcp_links
    traces, token = store.search_traces([experiment_id], filter_string=filter_string)
    assert [trace.trace_id for trace in traces] == expected_ids
    assert token is None


def test_search_traces_mcp_filters_before_pagination(store_with_mcp_links):
    store, experiment_id = store_with_mcp_links
    trace_ids = []
    token = None
    for _ in range(3):
        traces, token = store.search_traces(
            [experiment_id],
            filter_string="mcp_server = 'com.example/weather' AND tag.env = 'prod'",
            max_results=1,
            page_token=token,
        )
        trace_ids.extend(trace.trace_id for trace in traces)
        if token is None:
            break
    assert trace_ids == ["tr-a", "tr-b"]
    assert token is None


def test_search_traces_mcp_filters_respect_experiments(store_with_mcp_links):
    store, _ = store_with_mcp_links
    experiment_id = store.create_experiment("mcp-other-experiment")
    _create_trace(store, "tr-other", experiment_id=experiment_id)
    store.upsert_span_mcp_server_links(
        "tr-other", [MCPServerSpanObservation("s1", "com.example/weather")]
    )
    traces, _ = store.search_traces(
        [experiment_id], filter_string="mcp_server = 'com.example/weather'"
    )
    assert [trace.trace_id for trace in traces] == ["tr-other"]


@pytest.mark.parametrize(
    "dialect", [sqlite.dialect(), postgresql.dialect(), mysql.dialect(), mssql.dialect()]
)
def test_search_traces_mcp_filters_use_one_correlated_exists(store, dialect):
    with store.ManagedSessionMaker() as session:
        query = session.query(SqlTraceInfo)
        attribute_filters, non_attribute_filters, span_attribute_filters, span_filters, run_id = (
            _get_filter_clauses_for_search_traces(
                "mcp_server = 'com.example/weather' AND mcp_server.version = '1.0.0'",
                session,
                dialect.name,
                query,
            )
        )
        compiled_sql = str(
            query.filter(*attribute_filters).statement.compile(dialect=dialect)
        ).lower()
    assert compiled_sql.count("exists") == 1
    assert "span_mcp_server_links.trace_id = trace_info.request_id" in compiled_sql
    assert "experiments.experiment_id = trace_info.experiment_id" in compiled_sql
    assert "experiments.workspace = span_mcp_server_links.mcp_workspace" in compiled_sql
    assert "span_mcp_server_links.mcp_name" in compiled_sql
    assert "span_mcp_server_links.mcp_version" in compiled_sql
    assert "span_mcp_server_links" not in compiled_sql.split("exists")[0]
    assert non_attribute_filters == span_attribute_filters == span_filters == []
    assert run_id is None
