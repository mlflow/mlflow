import pytest
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite

from mlflow.entities import AssessmentSource, Feedback, TraceState
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import _get_filter_clauses_for_search_traces

from tests.store.tracking.sqlalchemy_store.conftest import _create_trace, create_test_span

pytestmark = pytest.mark.notrackingurimock


def test_search_trace_groups_preserve_scope_order_and_pagination(store):
    exp = store.create_experiment("grouped-search")
    other = store.create_experiment("other-experiment")
    for trace_id, experiment, timestamp, env, state in [
        ("tr-old-error", exp, 1, "prod", TraceState.ERROR),
        ("tr-new-error", exp, 20, "prod", TraceState.ERROR),
        ("tr-new-ok", exp, 30, "dev", TraceState.OK),
        ("tr-both", exp, 40, "prod", TraceState.ERROR),
        ("tr-other", other, 50, "prod", TraceState.ERROR),
    ]:
        _create_trace(
            store, trace_id, experiment, request_time=timestamp, state=state, tags={"env": env}
        )
    expression = "timestamp_ms > 10 AND (status = 'ERROR' OR tags.env = 'dev')"
    found = []
    token = None
    while True:
        rows, token = store.search_traces(
            [exp], filter_string=expression, max_results=1, page_token=token
        )
        found.extend(t.request_id for t in rows)
        if not token:
            break
    assert found == ["tr-both", "tr-new-ok", "tr-new-error"]
    rows, _ = store.search_traces(
        [exp], filter_string="status = 'ERROR' OR tags.env = 'dev' AND timestamp_ms > 10"
    )
    assert {t.request_id for t in rows} == {*found, "tr-old-error"}


def test_search_trace_groups_preserve_same_span_and_null_filters(store):
    exp = store.create_experiment("grouped-span-search")
    for trace_id in ["tr-split", "tr-same", "tr-alternative"]:
        _create_trace(store, trace_id, exp)
    store.log_spans(
        exp,
        [
            create_test_span("tr-split", name="target", span_type="CHAIN", span_id=1),
            create_test_span("tr-split", name="other", span_type="LLM", span_id=2),
            create_test_span("tr-same", name="target", span_type="LLM", span_id=3),
            create_test_span("tr-alternative", name="alternative", span_type="CHAIN", span_id=4),
        ],
    )
    rows, _ = store.search_traces(
        [exp],
        filter_string="(span.name = 'target' AND span.type = 'LLM') OR span.name = 'alternative'",
    )
    assert {t.request_id for t in rows} == {"tr-same", "tr-alternative"}
    rows, _ = store.search_traces(
        [exp],
        filter_string="tags.absent IS NULL AND (span.name = 'target' OR span.name = 'alternative')",
    )
    assert {t.request_id for t in rows} == {"tr-split", "tr-same", "tr-alternative"}


def test_search_trace_groups_validate_every_branch(store):
    exp = store.create_experiment("invalid-grouped-search")
    _create_trace(store, "tr-valid", exp)
    with pytest.raises(MlflowException, match="Invalid comparator"):
        store.search_traces([exp], filter_string="status = 'OK' OR tags.env > 'prod'")


def test_search_trace_groups_run_links_and_assessments(store):
    exp = store.create_experiment("grouped-linked-search")
    run = store.create_run(exp, user_id="user", start_time=0, tags=[], run_name="run")
    for trace_id in ["tr-linked", "tr-scored", "tr-none"]:
        _create_trace(store, trace_id, exp)
    store.link_traces_to_run(["tr-linked"], run.info.run_id)
    store.create_assessment(
        Feedback(
            trace_id="tr-scored",
            name="quality",
            value=0.8,
            source=AssessmentSource(source_type="HUMAN", source_id="user"),
        )
    )
    rows, _ = store.search_traces(
        [exp], filter_string=(f"run_id = '{run.info.run_id}' OR feedback.quality >= 0.5")
    )
    assert {t.request_id for t in rows} == {"tr-linked", "tr-scored"}


@pytest.mark.parametrize("dialect", [sqlite, postgresql, mysql, mssql])
def test_search_trace_groups_bind_values(store, dialect):
    marker = "literal OR (AND)"
    with store.ManagedSessionMaker(read_only=True) as session:
        query = store._trace_query(session)
        clauses = _get_filter_clauses_for_search_traces(
            f"status = 'OK' OR tags.env = '{marker}'", session, store._get_dialect(), query
        )
        statement = store._apply_trace_filter_clauses(query, *clauses).statement
        compiled = statement.compile(dialect=dialect.dialect())
        assert marker not in str(compiled)
        assert marker in compiled.params.values()


def test_search_trace_groups_keep_custom_span_equality_semantics(store):
    exp = store.create_experiment("grouped-custom-span-search")
    _create_trace(store, "tr-custom", exp)
    store.log_spans(exp, [create_test_span("tr-custom", attributes={"foo": "a"})])
    predicates = ["span.attributes.foo = 'a'", "span.attributes.foo = 'b'"]
    expected = {
        trace.request_id
        for predicate in predicates
        for trace in store.search_traces([exp], filter_string=predicate)[0]
    }
    rows, _ = store.search_traces([exp], filter_string=" OR ".join(predicates))
    assert {trace.request_id for trace in rows} == expected
