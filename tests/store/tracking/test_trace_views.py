import uuid
from pathlib import Path

import pytest

from mlflow.entities import TraceInfo, trace_location
from mlflow.entities.trace_info import TraceState
from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
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


def _make_ranges():
    return [
        SpanRange(
            from_selector=SpanSelector(span_name="AgentExecutor"),
            label="Summary",
            description="Agent ran and succeeded.",
        ),
        SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            to_selector=SpanSelector(span_type="TOOL"),
            label="Step 1",
            description="LLM called a tool.",
            input_path="$.reasoning",
            output_path="$.result",
        ),
    ]


def test_create_and_get_with_ranges(store, trace_id):
    view = TraceView(name="my-view", trace_id=trace_id, ranges=_make_ranges())
    created = store.create_trace_view(view)

    assert created.view_id is not None
    assert created.name == "my-view"
    assert len(created.ranges) == 2
    assert created.ranges[0].label == "Summary"
    assert created.ranges[0].range_id is not None
    assert created.ranges[1].label == "Step 1"
    assert created.ranges[1].to_selector.span_type == "TOOL"
    assert created.ranges[1].input_path == "$.reasoning"
    assert created.ranges[1].position == 1

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert len(fetched.ranges) == 2
    assert fetched.ranges[0].label == "Summary"
    assert fetched.ranges[1].output_path == "$.result"


def test_create_without_ranges(store, trace_id):
    view = TraceView(name="empty", trace_id=trace_id)
    created = store.create_trace_view(view)
    assert created.ranges == []

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert fetched.ranges == []


def test_create_experiment_scoped_with_ranges(store, experiment_id):
    view = TraceView(
        name="exp-view",
        experiment_id=experiment_id,
        ranges=[SpanRange(from_selector=SpanSelector(span_type="TOOL"), label="Tools")],
    )
    created = store.create_trace_view(view)
    assert created.experiment_id == experiment_id
    assert len(created.ranges) == 1


def test_list_views_returns_ranges(store, experiment_id, trace_id):
    store.create_trace_view(
        TraceView(name="v1", trace_id=trace_id, ranges=_make_ranges())
    )
    store.create_trace_view(
        TraceView(name="v2", experiment_id=experiment_id)
    )

    views = store.list_trace_views(trace_id=trace_id)
    names = {v.name for v in views}
    assert "v1" in names
    assert "v2" in names
    v1 = next(v for v in views if v.name == "v1")
    assert len(v1.ranges) == 2


def test_update_replaces_ranges(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="original", trace_id=trace_id, ranges=_make_ranges())
    )
    assert len(created.ranges) == 2

    new_ranges = [
        SpanRange(
            from_selector=SpanSelector(span_name="root"),
            label="New Summary",
            description="Updated summary.",
        ),
    ]
    updated = store.update_trace_view(
        view_id=created.view_id,
        ranges=new_ranges,
    )
    assert len(updated.ranges) == 1
    assert updated.ranges[0].label == "New Summary"

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert len(fetched.ranges) == 1


def test_update_name_only_preserves_ranges(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="original", trace_id=trace_id, ranges=_make_ranges())
    )

    updated = store.update_trace_view(
        view_id=created.view_id,
        name="renamed",
    )
    assert updated.name == "renamed"
    assert len(updated.ranges) == 2


def test_delete_cascades_ranges(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="to-delete", trace_id=trace_id, ranges=_make_ranges())
    )
    store.delete_trace_view(view_id=created.view_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_trace_view(trace_id=trace_id, view_id=created.view_id)


def test_create_invalid_scope_raises(store):
    view = TraceView(name="bad", trace_id="t1", experiment_id="e1")
    with pytest.raises(MlflowException, match="Exactly one of"):
        store.create_trace_view(view)


def test_get_nonexistent_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_trace_view(trace_id=None, view_id="tv-nonexistent")


def test_delete_nonexistent_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_trace_view(view_id="tv-nonexistent")
