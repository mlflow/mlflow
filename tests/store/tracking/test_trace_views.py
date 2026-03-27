import uuid
from pathlib import Path

import pytest

from mlflow.entities import TraceInfo, trace_location
from mlflow.entities.trace_info import TraceState
from mlflow.entities.trace_view import SpanFilter, TraceView
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis

DB_URI = "sqlite:///"


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    db_uri = f"{DB_URI}{tmp_path / 'test.db'}"
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store):
    return store.create_experiment("test-experiment")


@pytest.fixture
def trace_id(store, experiment_id):
    timestamp_ms = get_current_time_millis()
    tid = f"tr-{uuid.uuid4()}"
    trace_info = store.start_trace(
        TraceInfo(
            trace_id=tid,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=timestamp_ms,
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )
    return trace_info.trace_id


def test_create_and_get_trace_view_trace_scoped(store, trace_id):
    view = TraceView(name="my-view", trace_id=trace_id, description="A test view")
    created = store.create_trace_view(view)

    assert created.view_id is not None
    assert created.name == "my-view"
    assert created.trace_id == trace_id
    assert created.experiment_id is None
    assert created.description == "A test view"
    assert created.create_time_ms is not None
    assert created.last_update_time_ms is not None

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert fetched.view_id == created.view_id
    assert fetched.name == "my-view"


def test_create_and_get_trace_view_experiment_scoped(store, experiment_id):
    view = TraceView(name="exp-view", experiment_id=experiment_id)
    created = store.create_trace_view(view)

    assert created.view_id is not None
    assert created.experiment_id == experiment_id
    assert created.trace_id is None

    fetched = store.get_trace_view(trace_id=None, view_id=created.view_id)
    assert fetched.view_id == created.view_id
    assert fetched.name == "exp-view"


def test_create_trace_view_with_span_filter(store, trace_id):
    sf = SpanFilter(span_name="llm-call", span_type="LLM")
    view = TraceView(name="filtered", trace_id=trace_id, span_filter=sf)
    created = store.create_trace_view(view)

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert fetched.span_filter is not None
    assert fetched.span_filter.span_name == "llm-call"
    assert fetched.span_filter.span_type == "LLM"


def test_create_trace_view_invalid_scope_raises(store):
    view = TraceView(name="bad", trace_id="t1", experiment_id="e1")
    with pytest.raises(MlflowException, match="Exactly one of trace_id or experiment_id"):
        store.create_trace_view(view)


def test_get_nonexistent_trace_view_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_trace_view(trace_id=None, view_id="tv-nonexistent")


def test_list_trace_views_by_trace_id(store, experiment_id, trace_id):
    # Create a trace-scoped view
    store.create_trace_view(TraceView(name="trace-view", trace_id=trace_id))
    # Create an experiment-scoped view (should also appear for this trace)
    store.create_trace_view(TraceView(name="exp-view", experiment_id=experiment_id))

    views = store.list_trace_views(trace_id=trace_id)
    names = {v.name for v in views}
    assert "trace-view" in names
    assert "exp-view" in names
    assert len(views) == 2


def test_list_trace_views_by_experiment_id(store, experiment_id, trace_id):
    store.create_trace_view(TraceView(name="exp-view-1", experiment_id=experiment_id))
    store.create_trace_view(TraceView(name="exp-view-2", experiment_id=experiment_id))
    # Trace-scoped view should NOT appear when listing by experiment
    store.create_trace_view(TraceView(name="trace-only", trace_id=trace_id))

    views = store.list_trace_views(experiment_id=experiment_id)
    names = {v.name for v in views}
    assert "exp-view-1" in names
    assert "exp-view-2" in names
    assert "trace-only" not in names
    assert len(views) == 2


def test_update_trace_view(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="original", trace_id=trace_id, description="old")
    )

    updated = store.update_trace_view(
        view_id=created.view_id,
        name="renamed",
        description="new desc",
        span_filter=SpanFilter(span_name="updated-span"),
        input_path="$.input",
        output_path="$.output",
    )

    assert updated.name == "renamed"
    assert updated.description == "new desc"
    assert updated.span_filter.span_name == "updated-span"
    assert updated.input_path == "$.input"
    assert updated.output_path == "$.output"
    assert updated.last_update_time_ms >= created.last_update_time_ms

    # Verify persistence
    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert fetched.name == "renamed"


def test_update_trace_view_partial(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="keep-this", trace_id=trace_id, description="keep-desc")
    )

    updated = store.update_trace_view(
        view_id=created.view_id,
        description="changed-desc",
    )

    assert updated.name == "keep-this"
    assert updated.description == "changed-desc"


def test_update_nonexistent_trace_view_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_trace_view(view_id="tv-nonexistent", name="x")


def test_delete_trace_view(store, trace_id):
    created = store.create_trace_view(TraceView(name="to-delete", trace_id=trace_id))

    store.delete_trace_view(view_id=created.view_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_trace_view(trace_id=trace_id, view_id=created.view_id)


def test_delete_nonexistent_trace_view_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_trace_view(view_id="tv-nonexistent")
