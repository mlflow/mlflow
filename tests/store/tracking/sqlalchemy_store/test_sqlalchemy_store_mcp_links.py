from unittest import mock

import pytest
import sqlalchemy as sa
from opentelemetry.sdk.trace import TracerProvider
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite

from mlflow.entities import LiveSpan, Span
from mlflow.entities.trace_mcp_server_link import MCPServerSpanObservation
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import MSSQL, SQLITE
from mlflow.store.tracking.dbmodels.models import SqlSpan, SqlSpanMCPServerLink, SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import _get_filter_clauses_for_search_traces
from mlflow.tracing.constant import (
    MCP_SPAN_ATTRIBUTE_KEYS,
    SpanAttributeKey,
    SpansLocation,
    TraceTagKey,
)
from mlflow.tracing.otel.translation import strip_otlp_double_encoding
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.store.tracking.sqlalchemy_store.conftest import (
    _create_trace,
    create_test_span,
    upsert_mcp_server_links_for_test,
)
from tests.store.tracking.sqlalchemy_store.mcp_test_utils import get_mcp_server_links_for_test

pytestmark = pytest.mark.notrackingurimock


@pytest.mark.parametrize("protobuf_roundtrip", [False, True])
@pytest.mark.parametrize(
    "version",
    [
        "1",
        "true",
        "null",
        "[]",
        "{}",
        '"quoted"',
        "发布版",
        "🧪" * 256,
        1,
        True,
        " 1.0.0",
        "1.0.0 ",
        "\t1.0.0",
        "1.0.0\n",
    ],
)
def test_log_spans_preserves_native_otel_mcp_versions(store, version, protobuf_roundtrip):
    experiment_id = store.create_experiment("native-otel-mcp-version")
    with TracerProvider().get_tracer(__name__).start_as_current_span("mcp") as otel_span:
        span = LiveSpan(otel_span, trace_id="tr-native-mcp-version")
        otel_span.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, "com.example/weather")
        otel_span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, version)
    span = span.to_immutable_span()
    if protobuf_roundtrip:
        # Span.to_otel_proto is the MLflow LogSpans transport, not the raw OTLP endpoint.
        span = Span.from_otel_proto(span.to_otel_proto())
    store.log_spans(experiment_id, [span])

    expected_version = version if isinstance(version, str) else None
    assert get_mcp_server_links_for_test(store, span.trace_id) == [
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", expected_version)
    ]
    query_version = str(version)
    traces, _ = store.search_traces(
        [experiment_id],
        filter_string=(
            f"mcp_server = 'com.example/weather' AND mcp_server.version = '{query_version}'"
        ),
    )
    assert [trace.trace_id for trace in traces] == (
        [span.trace_id] if expected_version is not None else []
    )
    stored_version = (
        store
        .get_trace(span.trace_id)
        .data.spans[0]
        .get_attribute(SpanAttributeKey.MCP_SERVER_VERSION)
    )
    assert stored_version == version
    assert type(stored_version) is type(version)


@pytest.mark.parametrize("otlp", [False, True])
@pytest.mark.parametrize("value", [{"invalid": True}, [1, "invalid"], ["1.0.0"]])
def test_malformed_mcp_version_update_removes_stale_version(store, value, otlp):
    experiment_id = store.create_experiment("malformed-mcp-version-update")
    with TracerProvider().get_tracer(__name__).start_as_current_span("mcp") as otel_span:
        span = LiveSpan(otel_span, trace_id=f"tr-{otel_span.context.trace_id:032x}")
        span.set_attributes({
            SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
            SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0",
        })
        store.log_spans(experiment_id, [span.to_immutable_span()])
        assert get_mcp_server_links_for_test(store, span.trace_id) == [
            (DEFAULT_WORKSPACE_NAME, "com.example/weather", "1.0.0")
        ]
        span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, value)
    span = span.to_immutable_span()
    if otlp:
        span = Span.from_otel_proto(span.to_otel_proto())
        span._span._attributes = strip_otlp_double_encoding(
            span._span._attributes, raw_attribute_keys=MCP_SPAN_ATTRIBUTE_KEYS
        )
    store.log_spans(experiment_id, [span])

    assert get_mcp_server_links_for_test(store, span.trace_id) == [
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", None)
    ]
    traces, _ = store.search_traces(
        [experiment_id], filter_string="mcp_server = 'com.example/weather'"
    )
    assert [trace.trace_id for trace in traces] == [span.trace_id]
    traces, _ = store.search_traces([experiment_id], filter_string="mcp_server.version = '1.0.0'")
    assert traces == []
    assert (
        store
        .get_trace(span.trace_id)
        .data.spans[0]
        .get_attribute(SpanAttributeKey.MCP_SERVER_VERSION)
        is None
    )


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
    versions = ["1.0.0+Build", "1.0.0+build"]
    upsert_mcp_server_links_for_test(
        store,
        trace_id=trace.trace_id,
        links=[
            MCPServerSpanObservation(str(index), "com.example/weather", version)
            for index, version in enumerate([*versions, versions[0]])
        ],
    )
    summaries = get_mcp_server_links_for_test(store, trace_id=trace.trace_id)
    assert len(summaries) == len(versions)
    assert {summary[2] for summary in summaries} == set(versions)


def test_mcp_links_summarize_observations_without_spans_or_registry(store_and_trace_info):
    store, trace = store_and_trace_info
    assert get_mcp_server_links_for_test(store, trace.trace_id) == []
    links = [
        MCPServerSpanObservation("span-1", "com.example/weather", "1.0.0"),
        MCPServerSpanObservation("span-2", "com.example/weather", "1.0.0"),
        MCPServerSpanObservation("span-3", "com.example/weather"),
        MCPServerSpanObservation("span-4", "com.example/weather", "2.0.0"),
        MCPServerSpanObservation("span-1", "com.example/other"),
    ]
    upsert_mcp_server_links_for_test(store, trace.trace_id, links)
    upsert_mcp_server_links_for_test(store, trace.trace_id, links)

    assert set(get_mcp_server_links_for_test(store, trace.trace_id)) == {
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", "1.0.0"),
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", "2.0.0"),
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", None),
        (DEFAULT_WORKSPACE_NAME, "com.example/other", None),
    }
    with store.ManagedSessionMaker() as session:
        assert (
            session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace.trace_id).count()
        ) == len(links)


def test_mcp_link_names_are_case_sensitive_in_primary_key(store_and_trace_info):
    store, trace = store_and_trace_info
    names = ["com.example/Weather", "com.example/weather"]
    upsert_mcp_server_links_for_test(
        store,
        trace.trace_id,
        [MCPServerSpanObservation("span-1", name) for name in names],
    )

    assert set(get_mcp_server_links_for_test(store, trace.trace_id)) == {
        (DEFAULT_WORKSPACE_NAME, name, None) for name in names
    }


@pytest.mark.parametrize("version", ["2.0.0", None])
def test_mcp_links_update_version_preserving_creation_time(store_and_trace_info, version):
    store, trace = store_and_trace_info
    upsert_mcp_server_links_for_test(
        store, trace.trace_id, [MCPServerSpanObservation("span-1", "com.example/weather", "1.0.0")]
    )
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace.trace_id).update({
            "created_at": 123
        })

    upsert_mcp_server_links_for_test(
        store, trace.trace_id, [MCPServerSpanObservation("span-1", "com.example/weather", version)]
    )
    with store.ManagedSessionMaker() as session:
        row = session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace.trace_id).one()
        assert row.mcp_version == version
        assert row.created_at == 123


def test_mcp_links_duplicate_batch_keys_use_last_observation(store_and_trace_info):
    store, trace = store_and_trace_info
    upsert_mcp_server_links_for_test(
        store,
        trace.trace_id,
        [
            MCPServerSpanObservation("span-1", "com.example/weather", "1.0.0"),
            MCPServerSpanObservation("span-1", "com.example/weather", "2.0.0"),
        ],
    )
    assert get_mcp_server_links_for_test(store, trace.trace_id) == [
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", "2.0.0")
    ]


@pytest.mark.parametrize(
    "versions",
    [
        ("1.0.0+é", "1.0.0+e\u0301"),
        ("1.0.0+葛", "1.0.0+葛\U000e0100"),
        ("1.0.0+build", "1.0.0+build ", "1.0.0+build  "),
        ("发布🧪", "发布🧪 ", "发布🧪  "),
    ],
)
@pytest.mark.parametrize("source", ["spans", "store"])
def test_mcp_version_roundtrip_preserves_identity(store, versions, source):
    experiment_id = store.create_experiment("mcp-unicode-versions")
    if source == "spans":
        store.log_spans(
            experiment_id,
            [
                create_test_span(
                    f"tr-unicode-{index}",
                    attributes={
                        SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                        SpanAttributeKey.MCP_SERVER_VERSION: version,
                    },
                )
                for index, version in enumerate(versions)
            ],
        )
    else:
        for index, version in enumerate(versions):
            trace_id = f"tr-unicode-{index}"
            _create_trace(store, trace_id, experiment_id=experiment_id)
            upsert_mcp_server_links_for_test(
                store,
                trace_id,
                [MCPServerSpanObservation("span-1", "com.example/weather", version)],
            )
    if store.db_type in (MSSQL, SQLITE):
        # MySQL and PostgreSQL version equality follows the column collation, which may equate
        # distinct Unicode strings. Verify exact search identity only where the query enforces it.
        for index, version in enumerate(versions):
            traces, _ = store.search_traces(
                [experiment_id], filter_string=f"mcp_server.version = '{version}'"
            )
            assert [trace.trace_id for trace in traces] == [f"tr-unicode-{index}"]

    for index, version in enumerate(versions):
        assert get_mcp_server_links_for_test(store, f"tr-unicode-{index}") == [
            (DEFAULT_WORKSPACE_NAME, "com.example/weather", version)
        ]


@pytest.mark.parametrize(
    ("initial_version", "updated_version"),
    [(None, "1.0.0"), ("1.0.0", "2.0.0"), ("1.0.0", None)],
)
def test_log_spans_updates_mcp_version(store, initial_version, updated_version):
    experiment_id = store.create_experiment("mcp-version-update")
    for index, version in enumerate([initial_version, updated_version]):
        attributes = {SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"}
        if version is not None:
            attributes[SpanAttributeKey.MCP_SERVER_VERSION] = version
        span = create_test_span("trace-mcp-version-update", attributes=attributes)
        store.log_spans(experiment_id, [span])
        if index == 0:
            with store.ManagedSessionMaker(read_only=False) as session:
                session.query(SqlSpanMCPServerLink).filter_by(trace_id=span.trace_id).update({
                    "created_at": 123
                })
        assert get_mcp_server_links_for_test(store, span.trace_id) == [
            (DEFAULT_WORKSPACE_NAME, "com.example/weather", version)
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
        row = session.query(SqlSpanMCPServerLink).filter_by(trace_id=span.trace_id).one()
        assert row.created_at == 123
    assert (
        store
        .get_trace(span.trace_id)
        .data.spans[0]
        .get_attribute(SpanAttributeKey.MCP_SERVER_VERSION)
        == updated_version
    )


@pytest.mark.parametrize("updated_name", ["com.example/search", None, "https://example.com/mcp"])
def test_log_spans_replaces_mcp_links(store, updated_name):
    experiment_id = store.create_experiment("mcp-link-replacement")
    trace_id = "trace-mcp-replacement"
    store.log_spans(
        experiment_id,
        [
            create_test_span(
                trace_id,
                span_id=1,
                attributes={
                    SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                    SpanAttributeKey.MCP_SERVER_VERSION: "1.0.0",
                },
            ),
            create_test_span(
                trace_id,
                span_id=2,
                attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/other"},
            ),
            create_test_span(
                "trace-unrelated",
                span_id=1,
                attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"},
            ),
        ],
    )
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlSpanMCPServerLink).update({"created_at": 123})

    span = create_test_span(
        trace_id,
        span_id=1,
        attributes={SpanAttributeKey.MCP_SERVER_NAME: updated_name} if updated_name else {},
    )
    store.log_spans(experiment_id, [span])

    expected = {(DEFAULT_WORKSPACE_NAME, "com.example/other", None)}
    if updated_name == "com.example/search":
        expected.add((DEFAULT_WORKSPACE_NAME, updated_name, None))
    assert set(get_mcp_server_links_for_test(store, trace_id)) == expected
    traces, _ = store.search_traces(
        [experiment_id], filter_string="mcp_server = 'com.example/weather'"
    )
    assert [trace.trace_id for trace in traces] == ["trace-unrelated"]
    stored_spans = store.get_trace(trace_id).data.spans
    assert len(stored_spans) == 2
    assert (
        next(s for s in stored_spans if s.span_id == span.span_id).get_attribute(
            SpanAttributeKey.MCP_SERVER_NAME
        )
        == updated_name
    )
    with store.ManagedSessionMaker() as session:
        assert {
            row.created_at
            for row in session.query(SqlSpanMCPServerLink).filter(
                SqlSpanMCPServerLink.mcp_name.in_(["com.example/other", "com.example/weather"])
            )
        } == {123}


@pytest.mark.parametrize(
    ("initial_name", "updated_name"),
    [
        ("com.example/Weather", "com.example/weather"),
        ("com.example/weather", "com.example/Weather"),
    ],
)
def test_log_spans_replaces_mcp_links_on_case_only_rename(store, initial_name, updated_name):
    experiment_id = store.create_experiment("mcp-case-only-rename")
    trace_id = "trace-mcp-case-only-rename"
    for index, name in enumerate([initial_name, updated_name]):
        span = create_test_span(
            trace_id,
            attributes={
                SpanAttributeKey.MCP_SERVER_NAME: name,
                SpanAttributeKey.MCP_SERVER_VERSION: f"{index + 1}.0.0",
            },
        )
        store.log_spans(experiment_id, [span])
        if index == 0:
            with store.ManagedSessionMaker(read_only=False) as session:
                session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace_id).update({
                    "created_at": 123
                })

    assert get_mcp_server_links_for_test(store, trace_id) == [
        (DEFAULT_WORKSPACE_NAME, updated_name, "2.0.0")
    ]
    for name in [initial_name, updated_name]:
        traces, _ = store.search_traces([experiment_id], filter_string=f"mcp_server = '{name}'")
        assert [trace.trace_id for trace in traces] == ([trace_id] if name == updated_name else [])
    assert (
        store.get_trace(trace_id).data.spans[0].get_attribute(SpanAttributeKey.MCP_SERVER_NAME)
        == updated_name
    )
    with store.ManagedSessionMaker() as session:
        row = session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace_id).one()
        assert row.mcp_name == updated_name
        assert row.created_at != 123


def test_log_spans_reconciles_mcp_links_across_batches(store):
    experiment_id = store.create_experiment("mcp-reconciliation-batches")
    trace_id = "trace-mcp-reconciliation-batches"
    store.log_spans(
        experiment_id,
        [
            create_test_span(
                trace_id,
                span_id=index + 1,
                attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"},
            )
            for index in range(102)
        ],
    )
    store.log_spans(
        experiment_id,
        [
            create_test_span(
                trace_id,
                span_id=index + 1,
                attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/search"}
                if index % 2
                else {},
            )
            for index in range(102)
        ],
    )
    with store.ManagedSessionMaker() as session:
        rows = session.query(SqlSpanMCPServerLink).filter_by(trace_id=trace_id).all()
        assert {row.span_id for row in rows} == {f"{index:016x}" for index in range(2, 103, 2)}
        assert {row.mcp_name for row in rows} == {"com.example/search"}


@pytest.mark.parametrize(
    ("num_traces", "expected_batches"), [(1, 1), (50, 1), (100, 1), (101, 2), (250, 3)]
)
@pytest.mark.parametrize(
    ("initial_name", "updated_name"),
    [
        (None, None),
        ("com.example/weather", None),
        ("com.example/weather", "com.example/search"),
        ("com.example/weather", "com.example/weather"),
    ],
)
def test_log_spans_batches_mcp_link_writes_across_traces(
    store, num_traces, expected_batches, initial_name, updated_name
):
    experiment_id = store.create_experiment("mcp-link-write-query-count")
    statements = []

    def capture_statement(_conn, _cursor, statement, parameters, _context, _executemany):
        statements.append(statement.lower())
        if statement.lower().startswith("delete from span_mcp_server_links"):
            assert len(parameters) <= 999

    sa.event.listen(store.engine, "before_cursor_execute", capture_statement)
    try:
        for index, name in enumerate([initial_name, updated_name]):
            statements.clear()
            store.log_spans(
                experiment_id,
                [
                    create_test_span(
                        f"trace-mcp-cleanup-{index}",
                        attributes={SpanAttributeKey.MCP_SERVER_NAME: name} if name else {},
                    )
                    for index in range(num_traces)
                ],
            )
            if store.db_type in {"sqlite", "postgresql", "mysql"}:
                upserts = [
                    statement
                    for statement in statements
                    if statement.startswith("insert into span_mcp_server_links")
                ]
                assert len(upserts) == (expected_batches if name else 0)
            if index == 0:
                assert not any(
                    statement.startswith("delete from span_mcp_server_links")
                    for statement in statements
                )
    finally:
        sa.event.remove(store.engine, "before_cursor_execute", capture_statement)

    deletes = [
        statement
        for statement in statements
        if statement.startswith("delete from span_mcp_server_links")
    ]
    assert len(deletes) == expected_batches
    with store.ManagedSessionMaker() as session:
        rows = session.query(SqlSpanMCPServerLink).all()
        assert len(rows) == (num_traces if updated_name else 0)
        assert {row.mcp_name for row in rows} == ({updated_name} if updated_name else set())


@pytest.mark.parametrize("multiple_traces", [False, True])
@pytest.mark.parametrize("updated_name", [None, "com.example/search"])
def test_mcp_cleanup_does_not_scan_unrelated_links(store, multiple_traces, updated_name):
    if store.db_type != "sqlite":
        pytest.skip("Measures SQLite VM instructions without planner statistics")
    experiment_id = store.create_experiment("mcp-cleanup-index-lookup")
    spans = [
        create_test_span(f"trace-cleanup-{index if multiple_traces else 0}", span_id=index + 1)
        for index in range(50)
    ]
    store.log_spans(experiment_id, [*spans, create_test_span("trace-unrelated")])
    observations = {}
    for span in spans:
        observations.setdefault(span.trace_id, {})[span.span_id] = (
            MCPServerSpanObservation(span.span_id, updated_name) if updated_name else None
        )

    instruction_counts = []
    with store.ManagedSessionMaker(read_only=False) as session:
        connection = session.connection().connection.driver_connection
        for unrelated_count in [0, 5000]:
            if unrelated_count:
                session.execute(
                    sa.insert(SqlSpanMCPServerLink),
                    [
                        {
                            "trace_id": "trace-unrelated",
                            "span_id": f"{index:016x}",
                            "mcp_workspace": DEFAULT_WORKSPACE_NAME,
                            "mcp_name": "com.example/weather",
                        }
                        for index in range(unrelated_count)
                    ],
                )
            instructions = 0

            def count_instructions():
                nonlocal instructions
                instructions += 100
                return 0

            connection.set_progress_handler(count_instructions, 100)
            try:
                store._delete_stale_span_mcp_server_links(
                    session=session,
                    experiment_id=experiment_id,
                    observations_by_trace=observations,
                )
            finally:
                connection.set_progress_handler(None, 0)
            instruction_counts.append(instructions)
        assert session.query(SqlSpanMCPServerLink).count() == 5000

    # Index probes should not perform more work for every unrelated link in the workspace.
    assert instruction_counts[1] <= instruction_counts[0] + 1000


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
    assert set(get_mcp_server_links_for_test(store, trace_id)) == {
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", version) for version in [None, "1.0.0"]
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
        upsert_mcp_server_links_for_test(store, trace.trace_id, links)
    assert get_mcp_server_links_for_test(store, trace.trace_id) == []


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
    assert set(get_mcp_server_links_for_test(store, "trace-mcp")) == {
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", None),
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", "1.0.0+Build.1"),
    }
    assert get_mcp_server_links_for_test(store, "trace-other") == [
        (DEFAULT_WORKSPACE_NAME, "com.example/other", "latest")
    ]
    assert get_mcp_server_links_for_test(store, "trace-plain") == []
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
    expected = [(DEFAULT_WORKSPACE_NAME, expected_name, expected_version)] if expected_name else []
    assert get_mcp_server_links_for_test(store, span.trace_id) == expected
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
    assert get_mcp_server_links_for_test(store, span.trace_id) == [
        (DEFAULT_WORKSPACE_NAME, name, version)
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


@pytest.mark.parametrize("updated_name", ["com.example/search", None])
def test_log_spans_rolls_back_mcp_link_replacement(store, updated_name):
    experiment_id = store.create_experiment("atomic-mcp-replacement")
    trace_id = "trace-atomic-mcp-replacement"
    store.log_spans(
        experiment_id,
        [
            create_test_span(
                trace_id, attributes={SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"}
            )
        ],
    )
    delete_stale = store._delete_stale_span_mcp_server_links

    def delete_then_fail(**kwargs):
        delete_stale(**kwargs)
        raise RuntimeError("failure after MCP link deletion")

    with mock.patch.object(
        store, "_delete_stale_span_mcp_server_links", side_effect=delete_then_fail
    ) as deleted:
        with pytest.raises(MlflowException, match="failure after MCP link deletion"):
            store.log_spans(
                experiment_id,
                [
                    create_test_span(
                        trace_id,
                        attributes={SpanAttributeKey.MCP_SERVER_NAME: updated_name}
                        if updated_name
                        else {},
                    )
                ],
            )
        deleted.assert_called_once()

    assert (
        store.get_trace(trace_id).data.spans[0].get_attribute(SpanAttributeKey.MCP_SERVER_NAME)
        == "com.example/weather"
    )
    assert get_mcp_server_links_for_test(store, trace_id) == [
        (DEFAULT_WORKSPACE_NAME, "com.example/weather", None)
    ]


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
        upsert_mcp_server_links_for_test(store, trace_id, links)
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
    upsert_mcp_server_links_for_test(
        store, "tr-other", [MCPServerSpanObservation("s1", "com.example/weather")]
    )
    traces, _ = store.search_traces(
        [experiment_id], filter_string="mcp_server = 'com.example/weather'"
    )
    assert [trace.trace_id for trace in traces] == ["tr-other"]


@pytest.mark.parametrize(
    "dialect", [sqlite.dialect(), postgresql.dialect(), mysql.dialect(), mssql.dialect()]
)
@pytest.mark.parametrize("version", ["1.0.0", "发布🧪 "])
def test_search_traces_mcp_filters_use_one_correlated_exists(store, dialect, version):
    with store.ManagedSessionMaker() as session:
        query = session.query(SqlTraceInfo)
        attribute_filters, non_attribute_filters, span_attribute_filters, span_filters, run_id = (
            _get_filter_clauses_for_search_traces(
                f"mcp_server = 'com.example/weather' AND mcp_server.version = '{version}'",
                session,
                dialect.name,
                query,
            )
        )
        compiled = query.filter(*attribute_filters).statement.compile(dialect=dialect)
        compiled_sql = str(compiled).lower()
    assert compiled_sql.count("exists") == 1
    assert "span_mcp_server_links.trace_id = trace_info.request_id" in compiled_sql
    assert "experiments.experiment_id = trace_info.experiment_id" in compiled_sql
    assert "experiments.workspace = span_mcp_server_links.mcp_workspace" in compiled_sql
    assert "span_mcp_server_links.mcp_name" in compiled_sql
    assert "span_mcp_server_links.mcp_version" in compiled_sql
    if dialect.name == "mssql":
        assert "span_mcp_server_links.mcp_version collate latin1_general_100_bin2" in compiled_sql
        assert "datalength(span_mcp_server_links.mcp_version) = datalength(" in compiled_sql
        assert compiled_sql.count("datalength(") == 2
        version_binds = [bind for bind in compiled.binds.values() if bind.value == version]
        assert version_binds
        assert all(
            isinstance(bind.type.dialect_impl(dialect), sa.Unicode) for bind in version_binds
        )
    else:
        assert "datalength(" not in compiled_sql
    assert "span_mcp_server_links" not in compiled_sql.split("exists")[0]
    assert non_attribute_filters == span_attribute_filters == span_filters == []
    assert run_id is None
