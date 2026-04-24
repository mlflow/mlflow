import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

import mlflow
import mlflow.store.tracking.sqlalchemy_store as sqlalchemy_store_module
from mlflow.entities import (
    AssessmentSource,
    Expectation,
    ExperimentTag,
    Feedback,
    Link,
    trace_location,
)
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.entities.workspace import TraceArchivalConfig, Workspace
from mlflow.exceptions import (
    MlflowException,
    MlflowNotImplementedException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataNotFound,
    MlflowTracingException,
)
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    ErrorCode,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlSpan,
    SqlSpanMetrics,
    SqlTraceInfo,
    SqlTraceMetrics,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, _TraceArchiveCandidate
from mlflow.store.workspace.abstract_store import ResolvedTraceArchivalConfig
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    CostKey,
    SpanAttributeKey,
    SpansLocation,
    TraceArchivalFailureReason,
    TraceExperimentTagKey,
    TraceMetadataKey,
    TraceSizeStatsKey,
    TraceTagKey,
)
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.file_utils import TempDir, local_file_uri_to_path
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME, WORKSPACES_DIR_NAME

from tests.store.tracking.sqlalchemy_store.conftest import (
    IS_MSSQL,
    _create_experiments,
    _create_trace,
    create_mock_span_context,
    create_test_otel_span,
    create_test_span,
)

pytestmark = pytest.mark.notrackingurimock


def test_resolve_trace_archival_config_returns_defaults_for_single_tenant(
    store: SqlAlchemyStore, workspaces_enabled: bool
):
    if workspaces_enabled:
        pytest.skip("Workspace-aware resolution is covered separately.")

    resolved = store.resolve_trace_archival_config(
        default_trace_archival_location="s3://archive/default",
        default_retention="30d",
    )

    assert resolved.config == TraceArchivalConfig(
        location="s3://archive/default",
        retention="30d",
    )
    assert not resolved.append_workspace_prefix


def test_legacy_start_and_end_trace_v2(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_experiment")
    trace_info = store.deprecated_start_trace_v2(
        experiment_id=experiment_id,
        timestamp_ms=1234,
        request_metadata={"rq1": "foo", "rq2": "bar"},
        tags={"tag1": "apple", "tag2": "orange"},
    )
    request_id = trace_info.request_id

    assert trace_info.request_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1234
    assert trace_info.execution_time_ms is None
    assert trace_info.status == TraceStatus.IN_PROGRESS
    assert trace_info.request_metadata == {
        "rq1": "foo",
        "rq2": "bar",
    }
    artifact_location = trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
    assert artifact_location.endswith(f"/{experiment_id}/traces/{request_id}/artifacts")
    assert trace_info.tags == {
        "tag1": "apple",
        "tag2": "orange",
        MLFLOW_ARTIFACT_LOCATION: artifact_location,
    }
    assert trace_info.to_v3() == store.get_trace_info(request_id)

    trace_info = store.deprecated_end_trace_v2(
        request_id=request_id,
        timestamp_ms=2345,
        status=TraceStatus.OK,
        # Update one key and add a new key
        request_metadata={
            "rq1": "updated",
            "rq3": "baz",
        },
        tags={"tag1": "updated", "tag3": "grape"},
    )
    assert trace_info.request_id == request_id
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1234
    assert trace_info.execution_time_ms == 2345 - 1234
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        "rq1": "updated",
        "rq2": "bar",
        "rq3": "baz",
    }
    assert trace_info.tags == {
        "tag1": "updated",
        "tag2": "orange",
        "tag3": "grape",
        MLFLOW_ARTIFACT_LOCATION: artifact_location,
    }
    assert trace_info.to_v3() == store.get_trace_info(request_id)


def test_start_trace(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_experiment")
    trace_info = TraceInfo(
        trace_id="tr-123",
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.OK,
        tags={"tag1": "apple", "tag2": "orange"},
        trace_metadata={"rq1": "foo", "rq2": "bar"},
    )
    trace_info = store.start_trace(trace_info)
    trace_id = trace_info.trace_id

    assert trace_info.trace_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.request_time == 1234
    assert trace_info.execution_duration == 100
    assert trace_info.state == TraceState.OK
    assert {"rq1": "foo", "rq2": "bar"}.items() <= trace_info.trace_metadata.items()
    assert trace_info.trace_metadata.get(TraceMetadataKey.TRACE_INFO_FINALIZED) == "true"
    artifact_location = trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
    assert artifact_location.endswith(f"/{experiment_id}/traces/{trace_id}/artifacts")
    assert trace_info.tags == {
        "tag1": "apple",
        "tag2": "orange",
        MLFLOW_ARTIFACT_LOCATION: artifact_location,
    }
    assert trace_info == store.get_trace_info(trace_id)


@pytest.fixture
def store_with_traces(store):
    exp1 = store.create_experiment("exp1")
    exp2 = store.create_experiment("exp2")

    _create_trace(
        store,
        "tr-0",
        exp2,
        request_time=0,
        execution_duration=6,
        state=TraceState.OK,
        tags={"mlflow.traceName": "ddd"},
        trace_metadata={TraceMetadataKey.SOURCE_RUN: "run0"},
    )
    _create_trace(
        store,
        "tr-1",
        exp2,
        request_time=1,
        execution_duration=2,
        state=TraceState.ERROR,
        tags={"mlflow.traceName": "aaa", "fruit": "apple", "color": "red"},
        trace_metadata={TraceMetadataKey.SOURCE_RUN: "run1"},
    )
    _create_trace(
        store,
        "tr-2",
        exp1,
        request_time=2,
        execution_duration=4,
        state=TraceState.STATE_UNSPECIFIED,
        tags={"mlflow.traceName": "bbb", "fruit": "apple", "color": "green"},
    )
    _create_trace(
        store,
        "tr-3",
        exp1,
        request_time=3,
        execution_duration=10,
        state=TraceState.OK,
        tags={"mlflow.traceName": "ccc", "fruit": "orange"},
    )
    _create_trace(
        store,
        "tr-4",
        exp1,
        request_time=4,
        execution_duration=10,
        state=TraceState.OK,
        tags={"mlflow.traceName": "ddd", "color": "blue"},
    )

    return store


@pytest.mark.parametrize(
    ("order_by", "expected_ids"),
    [
        # Default order: descending by start time
        ([], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by start time
        (["timestamp"], ["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"]),
        (["timestamp DESC"], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by execution_time and timestamp
        (
            ["execution_time DESC", "timestamp ASC"],
            ["tr-3", "tr-4", "tr-0", "tr-2", "tr-1"],
        ),
        # Order by experiment ID
        (["experiment_id"], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by status
        (["status"], ["tr-1", "tr-4", "tr-3", "tr-0", "tr-2"]),
        # Order by name
        (["name"], ["tr-1", "tr-2", "tr-3", "tr-4", "tr-0"]),
        # Order by tag (null comes last)
        (["tag.fruit"], ["tr-2", "tr-1", "tr-3", "tr-4", "tr-0"]),
        # Order by multiple tags
        (["tag.fruit", "tag.color"], ["tr-2", "tr-1", "tr-3", "tr-4", "tr-0"]),
        # Order by non-existent tag (should be ordered by default order)
        (["tag.nonexistent"], ["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]),
        # Order by run Id
        (["run_id"], ["tr-0", "tr-1", "tr-4", "tr-3", "tr-2"]),
    ],
)
def test_search_traces_order_by(store_with_traces, order_by, expected_ids):
    exp1 = store_with_traces.get_experiment_by_name("exp1").experiment_id
    exp2 = store_with_traces.get_experiment_by_name("exp2").experiment_id
    trace_infos, _ = store_with_traces.search_traces(
        locations=[exp1, exp2],
        filter_string=None,
        max_results=5,
        order_by=order_by,
    )
    actual_ids = [trace_info.trace_id for trace_info in trace_infos]
    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ("filter_string", "expected_ids"),
    [
        # Search by name
        ("name = 'aaa'", ["tr-1"]),
        ("name != 'aaa'", ["tr-4", "tr-3", "tr-2", "tr-0"]),
        # Search by status
        ("status = 'OK'", ["tr-4", "tr-3", "tr-0"]),
        ("status != 'OK'", ["tr-2", "tr-1"]),
        ("attributes.status = 'OK'", ["tr-4", "tr-3", "tr-0"]),
        ("attributes.name != 'aaa'", ["tr-4", "tr-3", "tr-2", "tr-0"]),
        ("trace.status = 'OK'", ["tr-4", "tr-3", "tr-0"]),
        ("trace.name != 'aaa'", ["tr-4", "tr-3", "tr-2", "tr-0"]),
        # Search by timestamp
        ("`timestamp` >= 1 AND execution_time < 10", ["tr-2", "tr-1"]),
        # Search by tag
        ("tag.fruit = 'apple'", ["tr-2", "tr-1"]),
        # tags is an alias for tag
        ("tags.fruit = 'apple' and tags.color != 'red'", ["tr-2"]),
        # Search by request metadata
        ("run_id = 'run0'", ["tr-0"]),
        (f"request_metadata.{TraceMetadataKey.SOURCE_RUN} = 'run0'", ["tr-0"]),
        (f"request_metadata.{TraceMetadataKey.SOURCE_RUN} = 'run1'", ["tr-1"]),
        (f"request_metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run0'", ["tr-0"]),
        (f"metadata.{TraceMetadataKey.SOURCE_RUN} = 'run0'", ["tr-0"]),
        (f"metadata.{TraceMetadataKey.SOURCE_RUN} != 'run0'", ["tr-1"]),
    ],
)
def test_search_traces_with_filter(store_with_traces, filter_string, expected_ids):
    exp1 = store_with_traces.get_experiment_by_name("exp1").experiment_id
    exp2 = store_with_traces.get_experiment_by_name("exp2").experiment_id

    trace_infos, _ = store_with_traces.search_traces(
        locations=[exp1, exp2],
        filter_string=filter_string,
        max_results=5,
        order_by=[],
    )
    actual_ids = [trace_info.trace_id for trace_info in trace_infos]
    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ("filter_string", "error"),
    [
        ("invalid", r"Invalid clause\(s\) in filter string"),
        ("name = 'foo' AND invalid", r"Invalid clause\(s\) in filter string"),
        ("foo.bar = 'baz'", r"Invalid entity type 'foo'"),
        ("invalid = 'foo'", r"Invalid attribute key 'invalid'"),
        ("trace.tags.foo = 'bar'", r"Invalid attribute key 'tags\.foo'"),
        ("trace.status < 'OK'", r"Invalid comparator '<'"),
        ("name IN ('foo', 'bar')", r"Invalid comparator 'IN'"),
    ],
)
def test_search_traces_with_invalid_filter(store_with_traces, filter_string, error):
    exp1 = store_with_traces.get_experiment_by_name("exp1").experiment_id
    exp2 = store_with_traces.get_experiment_by_name("exp2").experiment_id

    with pytest.raises(MlflowException, match=error):
        store_with_traces.search_traces(
            locations=[exp1, exp2],
            filter_string=filter_string,
        )


def test_search_traces_raise_if_max_results_arg_is_invalid(store):
    with pytest.raises(
        MlflowException,
        match="Invalid value 50001 for parameter 'max_results' supplied.",
    ):
        store.search_traces(locations=[], max_results=50001)

    with pytest.raises(
        MlflowException, match="Invalid value -1 for parameter 'max_results' supplied."
    ):
        store.search_traces(locations=[], max_results=-1)


def test_search_traces_pagination(store_with_traces):
    exps = [
        store_with_traces.get_experiment_by_name("exp1").experiment_id,
        store_with_traces.get_experiment_by_name("exp2").experiment_id,
    ]

    traces, token = store_with_traces.search_traces(exps, max_results=2)
    assert [t.trace_id for t in traces] == ["tr-4", "tr-3"]

    traces, token = store_with_traces.search_traces(exps, max_results=2, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-2", "tr-1"]

    traces, token = store_with_traces.search_traces(exps, max_results=2, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-0"]
    assert token is None


def test_search_traces_pagination_tie_breaker(store):
    # This test is for ensuring the tie breaker for ordering traces with the same timestamp
    # works correctly.
    exp1 = store.create_experiment("exp1")

    trace_ids = [f"tr-{i}" for i in range(5)]
    random.shuffle(trace_ids)
    # Insert traces with random order
    for rid in trace_ids:
        _create_trace(store, rid, exp1, request_time=0)

    # Insert 5 more traces with newer timestamp
    trace_ids = [f"tr-{i + 5}" for i in range(5)]
    random.shuffle(trace_ids)
    for rid in trace_ids:
        _create_trace(store, rid, exp1, request_time=1)

    traces, token = store.search_traces([exp1], max_results=3)
    assert [t.trace_id for t in traces] == ["tr-5", "tr-6", "tr-7"]
    traces, token = store.search_traces([exp1], max_results=3, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-8", "tr-9", "tr-0"]
    traces, token = store.search_traces([exp1], max_results=3, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-1", "tr-2", "tr-3"]
    traces, token = store.search_traces([exp1], max_results=3, page_token=token)
    assert [t.trace_id for t in traces] == ["tr-4"]


def test_search_traces_with_run_id_filter(store: SqlAlchemyStore):
    # Create experiment and run
    exp_id = store.create_experiment("test_run_filter")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")
    run_id = run.info.run_id

    # Create traces with different relationships to the run
    # Trace 1: Has run_id in metadata (direct association)
    trace1_id = "tr-direct"
    _create_trace(store, trace1_id, exp_id, trace_metadata={"mlflow.sourceRun": run_id})

    # Trace 2: Linked via entity association
    trace2_id = "tr-linked"
    _create_trace(store, trace2_id, exp_id)
    store.link_traces_to_run([trace2_id], run_id)

    # Trace 3: Both metadata and entity association
    trace3_id = "tr-both"
    _create_trace(store, trace3_id, exp_id, trace_metadata={"mlflow.sourceRun": run_id})
    store.link_traces_to_run([trace3_id], run_id)

    # Trace 4: No association with the run
    trace4_id = "tr-unrelated"
    _create_trace(store, trace4_id, exp_id)

    # Search for traces with run_id filter
    traces, _ = store.search_traces([exp_id], filter_string=f'attributes.run_id = "{run_id}"')
    trace_ids = {t.trace_id for t in traces}

    # Should return traces 1, 2, and 3 but not 4
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test with another run to ensure isolation
    run2 = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run2")
    run2_id = run2.info.run_id

    # Create a trace linked to run2
    trace5_id = "tr-run2"
    _create_trace(store, trace5_id, exp_id)
    store.link_traces_to_run([trace5_id], run2_id)

    # Search for traces with run2_id filter
    traces, _ = store.search_traces([exp_id], filter_string=f'attributes.run_id = "{run2_id}"')
    trace_ids = {t.trace_id for t in traces}

    # Should only return trace5
    assert trace_ids == {trace5_id}

    # Original run_id search should still return the same traces
    traces, _ = store.search_traces([exp_id], filter_string=f'attributes.run_id = "{run_id}"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}


def test_search_traces_with_run_id_and_other_filters(store: SqlAlchemyStore):
    # Create experiment and run
    exp_id = store.create_experiment("test_combined_filters")
    run = store.create_run(exp_id, user_id="user", start_time=0, tags=[], run_name="test_run")
    run_id = run.info.run_id

    # Create traces with different tags and run associations
    trace1_id = "tr-tag1-linked"
    _create_trace(store, trace1_id, exp_id, tags={"type": "training"})
    store.link_traces_to_run([trace1_id], run_id)

    trace2_id = "tr-tag2-linked"
    _create_trace(store, trace2_id, exp_id, tags={"type": "inference"})
    store.link_traces_to_run([trace2_id], run_id)

    trace3_id = "tr-tag1-notlinked"
    _create_trace(store, trace3_id, exp_id, tags={"type": "training"})

    # Search with run_id and tag filter
    traces, _ = store.search_traces(
        [exp_id], filter_string=f'run_id = "{run_id}" AND tags.type = "training"'
    )
    trace_ids = {t.trace_id for t in traces}

    # Should only return trace1 (linked to run AND has training tag)
    assert trace_ids == {trace1_id}

    # Search with run_id only
    traces, _ = store.search_traces([exp_id], filter_string=f'run_id = "{run_id}"')
    trace_ids = {t.trace_id for t in traces}

    # Should return both linked traces
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_span_name_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_search")

    # Create traces with spans that have different names
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different names
    span1 = create_test_span(trace1_id, name="database_query", span_id=111, span_type="FUNCTION")
    span2 = create_test_span(trace2_id, name="api_call", span_id=222, span_type="FUNCTION")
    span3 = create_test_span(trace3_id, name="database_update", span_id=333, span_type="FUNCTION")

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test exact match
    traces, _ = store.search_traces([exp_id], filter_string='span.name = "database_query"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace1_id

    # Test LIKE pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "database%"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test match trace2 specifically
    traces, _ = store.search_traces([exp_id], filter_string='span.name = "api_call"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace2_id

    # Test NOT EQUAL
    traces, _ = store.search_traces([exp_id], filter_string='span.name != "api_call"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test no matches
    traces, _ = store.search_traces([exp_id], filter_string='span.name = "nonexistent"')
    assert len(traces) == 0


def test_search_traces_with_full_text_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_plain_text_search")

    # Create traces with spans that have different content
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different content
    span1 = create_test_span(
        trace1_id,
        name="database_query",
        span_id=111,
        span_type="FUNCTION",
        attributes={"llm.inputs": "what's MLflow?"},
    )
    span2 = create_test_span(
        trace2_id,
        name="api_request",
        span_id=222,
        span_type="TOOL",
        attributes={"response.token.usage": "123"},
    )
    span3 = create_test_span(
        trace3_id,
        name="computation",
        span_id=333,
        span_type="FUNCTION",
        attributes={"llm.outputs": 'MLflow is a tool for " testing " ...'},
    )
    span4 = create_test_span(
        trace3_id,
        name="result",
        span_id=444,
        parent_id=333,
        span_type="WORKFLOW",
        attributes={"test": '"the number increased 90%"'},
    )

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3, span4])

    # Test full text search using trace.text LIKE
    # match span name
    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%database_query%"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace1_id

    # match span type
    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%FUNCTION%"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # match span content / attributes
    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%what\'s MLflow?%"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.text LIKE "%MLflow is a tool for%"'
    )
    assert len(traces) == 1
    assert traces[0].trace_id == trace3_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%llm.%"')
    trace_ids = {t.trace_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    traces, _ = store.search_traces([exp_id], filter_string='trace.text LIKE "%90%%"')
    assert len(traces) == 1
    assert traces[0].trace_id == trace3_id


def test_search_traces_with_invalid_span_attribute(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_error")

    # Test invalid span attribute should raise error
    with pytest.raises(
        MlflowException,
        match=(
            "Invalid span attribute 'duration'. Supported attributes: name, status, "
            "type, attributes.<attribute_name>."
        ),
    ):
        store.search_traces([exp_id], filter_string='span.duration = "1000"')

    with pytest.raises(
        MlflowException,
        match=(
            "Invalid span attribute 'parent_id'. Supported attributes: name, status, "
            "type, attributes.<attribute_name>."
        ),
    ):
        store.search_traces([exp_id], filter_string='span.parent_id = "123"')

    with pytest.raises(
        MlflowException,
        match="span.content comparator '=' not one of ",
    ):
        store.search_traces([exp_id], filter_string='span.content = "test"')


def test_search_traces_with_span_type_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_type_search")

    # Create traces with spans that have different types
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different types
    span1 = create_test_span(trace1_id, name="llm_call", span_id=111, span_type="LLM")
    span2 = create_test_span(trace2_id, name="retriever_call", span_id=222, span_type="RETRIEVER")
    span3 = create_test_span(trace3_id, name="chain_call", span_id=333, span_type="CHAIN")

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test exact match
    traces, _ = store.search_traces([exp_id], filter_string='span.type = "LLM"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test IN operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type IN ("LLM", "RETRIEVER")')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test NOT IN operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type NOT IN ("LLM", "RETRIEVER")')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test != operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type != "LLM"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace2_id, trace3_id}

    # Test LIKE operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type LIKE "LLM"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test ILIKE operator
    traces, _ = store.search_traces([exp_id], filter_string='span.type ILIKE "llm"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_span_status_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_status_search")

    # Create traces with spans that have different statuses
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different statuses
    span1 = create_test_span(
        trace1_id, name="success_span", span_id=111, status=trace_api.StatusCode.OK
    )
    span2 = create_test_span(
        trace2_id, name="error_span", span_id=222, status=trace_api.StatusCode.ERROR
    )
    span3 = create_test_span(
        trace3_id, name="unset_span", span_id=333, status=trace_api.StatusCode.UNSET
    )

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test exact match with OK status
    traces, _ = store.search_traces([exp_id], filter_string='span.status = "OK"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test exact match with ERROR status
    traces, _ = store.search_traces([exp_id], filter_string='span.status = "ERROR"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test IN operator
    traces, _ = store.search_traces([exp_id], filter_string='span.status IN ("OK", "ERROR")')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test != operator
    traces, _ = store.search_traces([exp_id], filter_string='span.status != "ERROR"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace3_id}


def create_test_span_with_content(
    trace_id,
    name="test_span",
    span_id=111,
    parent_id=None,
    status=trace_api.StatusCode.UNSET,
    status_desc=None,
    start_ns=1000000000,
    end_ns=2000000000,
    span_type="LLM",
    trace_num=12345,
    custom_attributes=None,
    inputs=None,
    outputs=None,
) -> Span:
    context = create_mock_span_context(trace_num, span_id)
    parent_context = create_mock_span_context(trace_num, parent_id) if parent_id else None

    attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id),
        "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
    }

    # Add custom attributes
    if custom_attributes:
        for key, value in custom_attributes.items():
            attributes[key] = json.dumps(value, cls=TraceJSONEncoder)

    # Add inputs and outputs
    if inputs:
        attributes["mlflow.spanInputs"] = json.dumps(inputs, cls=TraceJSONEncoder)
    if outputs:
        attributes["mlflow.spanOutputs"] = json.dumps(outputs, cls=TraceJSONEncoder)

    otel_span = OTelReadableSpan(
        name=name,
        context=context,
        parent=parent_context,
        attributes=attributes,
        start_time=start_ns,
        end_time=end_ns,
        status=trace_api.Status(status, status_desc),
        resource=_OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def test_search_traces_with_span_content_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_content_search")

    # Create traces
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different content
    span1 = create_test_span_with_content(
        trace1_id,
        name="gpt_span",
        span_id=111,
        span_type="LLM",
        custom_attributes={"model": "gpt-4", "temperature": 0.7},
        inputs={"prompt": "Tell me about machine learning"},
        outputs={"response": "Machine learning is a subset of AI"},
    )

    span2 = create_test_span_with_content(
        trace2_id,
        name="claude_span",
        span_id=222,
        span_type="LLM",
        custom_attributes={"model": "claude-3", "max_tokens": 1000},
        inputs={"query": "What is neural network?"},
        outputs={"response": "A neural network is..."},
    )

    span3 = create_test_span_with_content(
        trace3_id,
        name="vector_span",
        span_id=333,
        span_type="RETRIEVER",
        custom_attributes={"database": "vector_store"},
        inputs={"search": "embeddings"},
        outputs={"documents": ["doc1", "doc2"]},
    )

    # Add spans to store
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test LIKE operator for model in content
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%gpt-4%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test LIKE operator for input text
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%neural network%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test LIKE operator for attribute
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%temperature%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test ILIKE operator (case-insensitive)
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.content ILIKE "%MACHINE LEARNING%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test LIKE with wildcard patterns
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%model%"')
    assert len(traces) == 2  # Both LLM spans have "model" in their attributes
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test searching for array content
    traces, _ = store.search_traces([exp_id], filter_string='span.content LIKE "%doc1%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


def test_search_traces_with_combined_span_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_combined_span_search")

    # Create traces
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create spans with various combinations
    span1 = create_test_span_with_content(
        trace1_id,
        name="llm_success",
        span_id=111,
        span_type="LLM",
        status=trace_api.StatusCode.OK,
        custom_attributes={"model": "gpt-4"},
    )

    span2 = create_test_span_with_content(
        trace2_id,
        name="llm_error",
        span_id=222,
        span_type="LLM",
        status=trace_api.StatusCode.ERROR,
        custom_attributes={"model": "gpt-3.5"},
    )

    span3 = create_test_span_with_content(
        trace3_id,
        name="retriever_success",
        span_id=333,
        span_type="RETRIEVER",
        status=trace_api.StatusCode.OK,
        custom_attributes={"database": "pinecone"},
    )

    span4 = create_test_span_with_content(
        trace4_id,
        name="llm_success_claude",
        span_id=444,
        span_type="LLM",
        status=trace_api.StatusCode.OK,
        custom_attributes={"model": "claude-3"},
    )

    # Add spans to store (must log spans for each trace separately)
    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])

    # Test: type = LLM AND status = OK
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type = "LLM" AND span.status = "OK"'
    )
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace4_id}

    # Test: type = LLM AND content contains gpt
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type = "LLM" AND span.content LIKE "%gpt%"'
    )
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    # Test: name LIKE pattern AND status = OK
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name LIKE "%success%" AND span.status = "OK"'
    )
    assert len(traces) == 3
    assert {t.request_id for t in traces} == {trace1_id, trace3_id, trace4_id}

    # Test: Complex combination - (type = LLM AND status = OK) AND content LIKE gpt
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='span.type = "LLM" AND span.status = "OK" AND span.content LIKE "%gpt-4%"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_combined_span_filters_match_same_span(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_same_span_filter")

    trace1_id = "trace1"
    _create_trace(store, trace1_id, exp_id)

    span1a = create_test_span_with_content(
        trace1_id,
        name="search_web",
        span_id=111,
        span_type="TOOL",
        status=trace_api.StatusCode.ERROR,
        custom_attributes={"query": "test"},
    )
    span1b = create_test_span_with_content(
        trace1_id,
        name="other_tool",
        span_id=112,
        span_type="TOOL",
        status=trace_api.StatusCode.OK,
        custom_attributes={"data": "value"},
    )

    trace2_id = "trace2"
    _create_trace(store, trace2_id, exp_id)

    span2 = create_test_span_with_content(
        trace2_id,
        name="search_web",
        span_id=222,
        span_type="TOOL",
        status=trace_api.StatusCode.OK,
        custom_attributes={"query": "test2"},
    )

    store.log_spans(exp_id, [span1a, span1b])
    store.log_spans(exp_id, [span2])

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name = "search_web" AND span.status = "OK"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name = "search_web" AND span.status = "ERROR"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name = "other_tool" AND span.status = "OK"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces([exp_id], filter_string='span.name = "search_web"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.status = "OK"')
    assert len(traces) == 2
    assert {t.request_id for t in traces} == {trace1_id, trace2_id}


def test_search_traces_span_filters_with_no_results(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_no_results")

    # Create a trace with a span
    trace_id = "trace1"
    _create_trace(store, trace_id, exp_id)

    span = create_test_span_with_content(
        trace_id,
        name="test_span",
        span_id=111,
        span_type="LLM",
        status=trace_api.StatusCode.OK,
        custom_attributes={"model": "gpt-4"},
    )

    store.log_spans(exp_id, [span])

    # Test searching for non-existent type
    traces, _ = store.search_traces([exp_id], filter_string='span.type = "NONEXISTENT"')
    assert len(traces) == 0

    # Test searching for non-existent content
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.content LIKE "%nonexistent_model%"'
    )
    assert len(traces) == 0

    # Test contradictory conditions
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type = "LLM" AND span.type = "RETRIEVER"'
    )
    assert len(traces) == 0


def test_search_traces_with_span_attributes_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_attributes_search")

    # Create traces with spans having custom attributes
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different custom attributes
    span1 = create_test_span_with_content(
        trace1_id,
        name="llm_span",
        span_id=111,
        span_type="LLM",
        custom_attributes={"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000},
    )

    span2 = create_test_span_with_content(
        trace2_id,
        name="llm_span",
        span_id=222,
        span_type="LLM",
        custom_attributes={"model": "claude-3", "temperature": 0.5, "provider": "anthropic"},
    )

    span3 = create_test_span_with_content(
        trace3_id,
        name="retriever_span",
        span_id=333,
        span_type="RETRIEVER",
        custom_attributes={"database": "pinecone", "top_k": 10, "similarity.threshold": 0.8},
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model LIKE "%gpt-4%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.temperature LIKE "%0.7%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.provider LIKE "%anthropic%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.database LIKE "%pinecone%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.nonexistent LIKE "%value%"'
    )
    assert len(traces) == 0

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.similarity.threshold LIKE "%0.8%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


def test_search_traces_with_feedback_and_expectation_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_feedback_expectation_search")

    # Create multiple traces
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create feedback for trace1 and trace2
    feedback1 = Feedback(
        trace_id=trace1_id,
        name="correctness",
        value=True,
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
        rationale="The response is accurate",
    )

    feedback2 = Feedback(
        trace_id=trace2_id,
        name="correctness",
        value=False,
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gpt-4"),
        rationale="The response contains errors",
    )

    feedback3 = Feedback(
        trace_id=trace2_id,
        name="helpfulness",
        value=5,
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    feedback4 = Feedback(
        trace_id=trace1_id,
        name="quality",
        value="high",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )

    # Create expectations for trace3 and trace4
    expectation1 = Expectation(
        trace_id=trace3_id,
        name="response_length",
        value=150,
        source=AssessmentSource(source_type="CODE", source_id="length_checker"),
    )

    expectation2 = Expectation(
        trace_id=trace4_id,
        name="response_length",
        value=200,
        source=AssessmentSource(source_type="CODE", source_id="length_checker"),
    )

    expectation3 = Expectation(
        trace_id=trace4_id,
        name="latency_ms",
        value=1000,
        source=AssessmentSource(source_type="CODE", source_id="latency_monitor"),
    )

    expectation4 = Expectation(
        trace_id=trace3_id,
        name="priority",
        value="urgent",
        source=AssessmentSource(source_type="CODE", source_id="priority_checker"),
    )

    # Store assessments
    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)
    store.create_assessment(feedback4)
    store.create_assessment(expectation1)
    store.create_assessment(expectation2)
    store.create_assessment(expectation3)
    store.create_assessment(expectation4)

    # Test: Search for traces with correctness feedback = True
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "true"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Search for traces with correctness feedback = False
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "false"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Search for traces with helpfulness feedback = 5
    traces, _ = store.search_traces([exp_id], filter_string='feedback.helpfulness = "5"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Search for traces with string-valued feedback
    traces, _ = store.search_traces([exp_id], filter_string='feedback.quality = "high"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Search for traces with response_length expectation = 150
    traces, _ = store.search_traces([exp_id], filter_string='expectation.response_length = "150"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Search for traces with response_length expectation = 200
    traces, _ = store.search_traces([exp_id], filter_string='expectation.response_length = "200"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: Search for traces with latency_ms expectation = 1000
    traces, _ = store.search_traces([exp_id], filter_string='expectation.latency_ms = "1000"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: Search for traces with string-valued expectation
    traces, _ = store.search_traces([exp_id], filter_string='expectation.priority = "urgent"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Combined filter with AND - trace with multiple expectations
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='expectation.response_length = "200" AND expectation.latency_ms = "1000"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: Search for non-existent feedback
    traces, _ = store.search_traces([exp_id], filter_string='feedback.nonexistent = "value"')
    assert len(traces) == 0

    # Test: Search for non-existent expectation
    traces, _ = store.search_traces([exp_id], filter_string='expectation.nonexistent = "value"')
    assert len(traces) == 0


def test_search_traces_with_run_id(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_run_id")
    run1_id = "run1"
    run2_id = "run2"
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"mlflow.sourceRun": run1_id})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"mlflow.sourceRun": run2_id})
    _create_trace(store, trace3_id, exp_id)

    traces, _ = store.search_traces([exp_id], filter_string='trace.run_id = "run1"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.run_id = "run2"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.run_id = "run3"')
    assert len(traces) == 0


def test_search_traces_with_client_request_id_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_client_request_id")

    # Create traces with different client_request_ids
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, client_request_id="client-req-abc")
    _create_trace(store, trace2_id, exp_id, client_request_id="client-req-xyz")
    _create_trace(store, trace3_id, exp_id, client_request_id=None)

    # Test: Exact match with =
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id = "client-req-abc"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Not equal with !=
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id != "client-req-abc"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id LIKE "%abc%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: ILIKE case-insensitive pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id ILIKE "%ABC%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_name_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_name_like")

    # Create traces with different names
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateResponse"})
    _create_trace(store, trace2_id, exp_id, tags={TraceTagKey.TRACE_NAME: "QueryDatabase"})
    _create_trace(store, trace3_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateEmbedding"})

    # Test: LIKE with prefix
    traces, _ = store.search_traces([exp_id], filter_string='trace.name LIKE "Generate%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='trace.name LIKE "%Database"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "%response%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: ILIKE with wildcard in middle
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "%generate%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}


def test_search_traces_with_tag_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_like")

    # Create traces with different tag values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={"environment": "production-us-east-1"})
    _create_trace(store, trace2_id, exp_id, tags={"environment": "production-us-west-2"})
    _create_trace(store, trace3_id, exp_id, tags={"environment": "staging-us-east-1"})

    # Test: LIKE with prefix
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment LIKE "production%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment LIKE "%-us-east-1"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment ILIKE "%PRODUCTION%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_feedback_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_feedback_like")

    # Create traces with different feedback values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create feedback with string values that can be pattern matched
    feedback1 = Feedback(
        trace_id=trace1_id,
        name="comment",
        value="Great response! Very helpful.",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )

    feedback2 = Feedback(
        trace_id=trace2_id,
        name="comment",
        value="Response was okay but could be better.",
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    feedback3 = Feedback(
        trace_id=trace3_id,
        name="comment",
        value="Not helpful at all.",
        source=AssessmentSource(source_type="HUMAN", source_id="user3@example.com"),
    )

    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)

    # Test: LIKE pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment LIKE "%helpful%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive pattern matching
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment ILIKE "%GREAT%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: LIKE with negation - response was okay
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment LIKE "%okay%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


def test_search_traces_with_assessment_is_null_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_null_filters")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"
    trace5_id = "trace5"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)
    _create_trace(store, trace5_id, exp_id)

    feedback1 = Feedback(
        trace_id=trace1_id,
        name="quality",
        value="good",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )
    feedback2 = Feedback(
        trace_id=trace2_id,
        name="quality",
        value="bad",
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    expectation1 = Expectation(
        trace_id=trace4_id,
        name="score",
        value=85,
        source=AssessmentSource(source_type="CODE", source_id="scorer"),
    )

    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(expectation1)

    traces, _ = store.search_traces([exp_id], filter_string="feedback.quality IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string="feedback.quality IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id, trace5_id}

    traces, _ = store.search_traces([exp_id], filter_string="expectation.score IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace4_id}

    traces, _ = store.search_traces([exp_id], filter_string="expectation.score IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id, trace5_id}

    traces, _ = store.search_traces(
        [exp_id],
        filter_string='feedback.quality IS NOT NULL AND feedback.quality = "good"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_feedback_filters_excludes_invalid_assessments(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("test_feedback_filters_excludes_invalid")

    trace1_id = "trace1"
    trace2_id = "trace2"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)

    # trace1: overridden from "no" to "yes" - only "yes" is valid
    original_feedback = Feedback(
        trace_id=trace1_id,
        name="correctness",
        value="no",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
    )
    created_original = store.create_assessment(original_feedback)

    override_feedback = Feedback(
        trace_id=trace1_id,
        name="correctness",
        value="yes",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
        overrides=created_original.assessment_id,
    )
    store.create_assessment(override_feedback)

    # trace2: "no" assessment, never overridden
    feedback2 = Feedback(
        trace_id=trace2_id,
        name="correctness",
        value="no",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
    )
    store.create_assessment(feedback2)

    # Filtering by "yes" should return only trace1 (current valid assessment)
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "yes"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Filtering by "no" should return only trace2 (trace1's "no" is invalid/overridden)
    traces, _ = store.search_traces([exp_id], filter_string='feedback.correctness = "no"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # IS NOT NULL should return both (both have a valid assessment)
    traces, _ = store.search_traces([exp_id], filter_string="feedback.correctness IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_session_scoped_assessment_expands_to_all_session_traces(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("test_session_assessment_expansion")

    # Session A: 3 traces
    _create_trace(
        store, "sa-t1", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"}
    )
    _create_trace(
        store, "sa-t2", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"}
    )
    _create_trace(
        store, "sa-t3", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"}
    )

    # Session B: 3 traces
    _create_trace(
        store, "sb-t1", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"}
    )
    _create_trace(
        store, "sb-t2", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"}
    )
    _create_trace(
        store, "sb-t3", exp_id, trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"}
    )

    # Add a session-scoped assessment on the first trace of session A
    session_feedback = Feedback(
        trace_id="sa-t1",
        name="session_quality",
        value="good",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
        metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
    )
    store.create_assessment(session_feedback)

    # Add a non-session assessment on a trace in session B (no session metadata)
    non_session_feedback = Feedback(
        trace_id="sb-t1",
        name="trace_quality",
        value="bad",
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
    )
    store.create_assessment(non_session_feedback)

    # Searching with the session-scoped assessment filter should return all 3 traces
    # from session A (not just the one with the assessment)
    traces, _ = store.search_traces([exp_id], filter_string='feedback.session_quality = "good"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sa-t1", "sa-t2", "sa-t3"}

    # IS NOT NULL with session-scoped assessment should also expand
    traces, _ = store.search_traces([exp_id], filter_string="feedback.session_quality IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sa-t1", "sa-t2", "sa-t3"}

    # Searching with non-session assessment should return only the single matching trace
    traces, _ = store.search_traces([exp_id], filter_string='feedback.trace_quality = "bad"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sb-t1"}

    # IS NOT NULL with non-session assessment should return only the single trace
    traces, _ = store.search_traces([exp_id], filter_string="feedback.trace_quality IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {"sb-t1"}

    # IS NULL with session-scoped assessment should exclude all session siblings too
    traces, _ = store.search_traces([exp_id], filter_string="feedback.session_quality IS NULL")
    trace_ids = {t.request_id for t in traces}
    # All session-A traces are excluded (session-scoped assessment covers the whole session)
    assert trace_ids == {"sb-t1", "sb-t2", "sb-t3"}


def test_search_traces_with_expectation_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_expectation_like")

    # Create traces with different expectation values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create expectations with string values
    expectation1 = Expectation(
        trace_id=trace1_id,
        name="output_format",
        value="JSON with nested structure",
        source=AssessmentSource(source_type="CODE", source_id="validator"),
    )

    expectation2 = Expectation(
        trace_id=trace2_id,
        name="output_format",
        value="XML document",
        source=AssessmentSource(source_type="CODE", source_id="validator"),
    )

    expectation3 = Expectation(
        trace_id=trace3_id,
        name="output_format",
        value="JSON array",
        source=AssessmentSource(source_type="CODE", source_id="validator"),
    )

    store.create_assessment(expectation1)
    store.create_assessment(expectation2)
    store.create_assessment(expectation3)

    # Test: LIKE pattern matching
    traces, _ = store.search_traces(
        [exp_id], filter_string='expectation.output_format LIKE "%JSON%"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces(
        [exp_id], filter_string='expectation.output_format ILIKE "%xml%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE with specific pattern
    traces, _ = store.search_traces(
        [exp_id], filter_string='expectation.output_format LIKE "%nested%"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id


def test_search_traces_with_metadata_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_like")

    # Create traces with different metadata values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(
        store, trace1_id, exp_id, trace_metadata={"custom_field": "production-deployment-v1"}
    )
    _create_trace(
        store, trace2_id, exp_id, trace_metadata={"custom_field": "production-deployment-v2"}
    )
    _create_trace(
        store, trace3_id, exp_id, trace_metadata={"custom_field": "staging-deployment-v1"}
    )

    # Test: LIKE with prefix
    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.custom_field LIKE "production%"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='metadata.custom_field LIKE "%-v1"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.custom_field ILIKE "%PRODUCTION%"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_combined_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_combined_filters")

    # Create traces with various attributes
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(
        store,
        trace1_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "GenerateResponse", "env": "production"},
        client_request_id="req-prod-001",
    )
    _create_trace(
        store,
        trace2_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "GenerateResponse", "env": "staging"},
        client_request_id="req-staging-001",
    )
    _create_trace(
        store,
        trace3_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "QueryDatabase", "env": "production"},
        client_request_id="req-prod-002",
    )
    _create_trace(
        store,
        trace4_id,
        exp_id,
        tags={TraceTagKey.TRACE_NAME: "QueryDatabase", "env": "staging"},
        client_request_id="req-staging-002",
    )

    # Test: Combine LIKE filters with AND
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.name LIKE "Generate%" AND tag.env = "production"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Combine ILIKE with exact match
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.client_request_id ILIKE "%PROD%" AND trace.name = "QueryDatabase"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Multiple LIKE conditions
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.name LIKE "%Response%" AND trace.client_request_id LIKE "%-staging-%"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: ILIKE on tag with exact match on another field
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='tag.env ILIKE "%STAGING%" AND trace.name != "GenerateResponse"',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


def test_search_traces_with_client_request_id_edge_cases(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_client_request_id_edge")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    # Various client_request_id formats
    _create_trace(store, trace1_id, exp_id, client_request_id="simple")
    _create_trace(store, trace2_id, exp_id, client_request_id="with-dashes-123")
    _create_trace(store, trace3_id, exp_id, client_request_id="WITH_UNDERSCORES_456")
    _create_trace(store, trace4_id, exp_id, client_request_id=None)

    # Test: LIKE with wildcard at start
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id LIKE "%123"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE with wildcard at end
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id LIKE "WITH%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: ILIKE finds case-insensitive match
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id ILIKE "with%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: Exact match still works
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id = "simple"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: != excludes matched trace
    traces, _ = store.search_traces([exp_id], filter_string='trace.client_request_id != "simple"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}


def test_search_traces_with_name_ilike_variations(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_name_ilike_variations")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, tags={TraceTagKey.TRACE_NAME: "USER_LOGIN"})
    _create_trace(store, trace2_id, exp_id, tags={TraceTagKey.TRACE_NAME: "user_logout"})
    _create_trace(store, trace3_id, exp_id, tags={TraceTagKey.TRACE_NAME: "User_Profile_Update"})
    _create_trace(store, trace4_id, exp_id, tags={TraceTagKey.TRACE_NAME: "AdminDashboard"})

    # Test: ILIKE finds all user-related traces regardless of case
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "user%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: ILIKE with wildcard in middle
    traces, _ = store.search_traces([exp_id], filter_string='trace.name ILIKE "%_log%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE is case-sensitive (should not match)
    traces, _ = store.search_traces([exp_id], filter_string='trace.name LIKE "user%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id}  # Only lowercase match

    # Test: Exact match with !=
    traces, _ = store.search_traces([exp_id], filter_string='trace.name != "USER_LOGIN"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}


def test_search_traces_with_span_name_like_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_name_like")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create spans with different names
    span1 = create_test_span_with_content(
        trace1_id, name="llm.generate_response", span_id=111, span_type="LLM"
    )
    span2 = create_test_span_with_content(
        trace2_id, name="llm.generate_embedding", span_id=222, span_type="LLM"
    )
    span3 = create_test_span_with_content(
        trace3_id, name="database.query_users", span_id=333, span_type="TOOL"
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])

    # Test: LIKE with prefix
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "llm.%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with suffix
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "%_response"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: ILIKE case-insensitive
    traces, _ = store.search_traces([exp_id], filter_string='span.name ILIKE "%GENERATE%"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: LIKE with wildcard in middle
    traces, _ = store.search_traces([exp_id], filter_string='span.name LIKE "%base.%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_name_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_name_rlike")

    # Create traces with different names
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateResponse"})
    _create_trace(store, trace2_id, exp_id, tags={TraceTagKey.TRACE_NAME: "QueryDatabase"})
    _create_trace(store, trace3_id, exp_id, tags={TraceTagKey.TRACE_NAME: "GenerateEmbedding"})
    _create_trace(store, trace4_id, exp_id, tags={TraceTagKey.TRACE_NAME: "api_v1_call"})

    # Test: RLIKE with regex pattern matching "Generate" at start
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "^Generate"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: RLIKE with regex pattern matching "Database" at end
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "Database$"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: RLIKE with character class [RE]
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "^Generate[RE]"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id}

    # Test: RLIKE with alternation (OR)
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "(Query|Embedding)"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: RLIKE with digit pattern
    traces, _ = store.search_traces([exp_id], filter_string='trace.name RLIKE "v[0-9]+"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_tag_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_rlike")

    # Create traces with different tag values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, tags={"environment": "production-us-east-1"})
    _create_trace(store, trace2_id, exp_id, tags={"environment": "production-us-west-2"})
    _create_trace(store, trace3_id, exp_id, tags={"environment": "staging-us-east-1"})
    _create_trace(store, trace4_id, exp_id, tags={"environment": "dev-local"})

    # Test: RLIKE with regex pattern for production environments
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment RLIKE "^production"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with pattern matching regions
    traces, _ = store.search_traces(
        [exp_id], filter_string='tag.environment RLIKE "us-(east|west)-[0-9]"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: RLIKE with negation pattern (not starting with production/staging)
    traces, _ = store.search_traces([exp_id], filter_string='tag.environment RLIKE "^dev"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_span_name_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_name_rlike")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"
    trace5_id = "trace5"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)
    _create_trace(store, trace5_id, exp_id)

    # Create spans with different names
    span1 = create_test_span_with_content(
        trace1_id, name="llm.generate_response", span_id=111, span_type="LLM"
    )
    span2 = create_test_span_with_content(
        trace2_id, name="llm.generate_embedding", span_id=222, span_type="LLM"
    )
    span3 = create_test_span_with_content(
        trace3_id, name="database.query_users", span_id=333, span_type="TOOL"
    )
    span4 = create_test_span_with_content(
        trace4_id, name="api_v2_endpoint", span_id=444, span_type="TOOL"
    )
    span5 = create_test_span_with_content(
        trace5_id, name="base.query_users", span_id=444, span_type="TOOL"
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])
    store.log_spans(exp_id, [span5])

    # Test: RLIKE with pattern matching llm namespace
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "^llm\\."')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with alternation for different operations
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "(response|users)"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id, trace5_id}

    # Test: RLIKE with version pattern
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "v[0-9]+_"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: RLIKE matching embedded substring
    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "query"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace5_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.name RLIKE "query_users"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace5_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.name RLIKE "^database\\.query_users$"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_feedback_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_feedback_rlike")

    # Create traces with different feedback values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)

    # Create feedback with string values that can be pattern matched
    from mlflow.entities.assessment import AssessmentSource, Feedback

    feedback1 = Feedback(
        trace_id=trace1_id,
        name="comment",
        value="Great response! Very helpful.",
        source=AssessmentSource(source_type="HUMAN", source_id="user1@example.com"),
    )

    feedback2 = Feedback(
        trace_id=trace2_id,
        name="comment",
        value="Response was okay but could be better.",
        source=AssessmentSource(source_type="HUMAN", source_id="user2@example.com"),
    )

    feedback3 = Feedback(
        trace_id=trace3_id,
        name="comment",
        value="Not helpful at all.",
        source=AssessmentSource(source_type="HUMAN", source_id="user3@example.com"),
    )

    store.create_assessment(feedback1)
    store.create_assessment(feedback2)
    store.create_assessment(feedback3)

    # Test: RLIKE pattern matching response patterns
    traces, _ = store.search_traces(
        [exp_id], filter_string='feedback.comment RLIKE "Great.*helpful"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: RLIKE with alternation
    traces, _ = store.search_traces(
        [exp_id], filter_string='feedback.comment RLIKE "(okay|better)"'
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: RLIKE matching negative feedback
    traces, _ = store.search_traces([exp_id], filter_string='feedback.comment RLIKE "Not.*all"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id


def test_search_traces_with_metadata_is_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_is_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, trace_metadata={})

    traces, _ = store.search_traces([exp_id], filter_string="metadata.region IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    traces, _ = store.search_traces([exp_id], filter_string="metadata.env IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.region IS NULL AND metadata.env = "staging"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id}


def test_search_traces_with_metadata_is_not_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_is_not_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, trace_metadata={})

    traces, _ = store.search_traces([exp_id], filter_string="metadata.region IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}

    traces, _ = store.search_traces([exp_id], filter_string="metadata.env IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.region IS NOT NULL AND metadata.env = "production"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}


def test_search_traces_with_tag_is_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_is_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, tags={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, tags={})

    traces, _ = store.search_traces([exp_id], filter_string="tag.region IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    traces, _ = store.search_traces([exp_id], filter_string="tag.env IS NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='tag.region IS NULL AND tag.env = "staging"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id}


def test_search_traces_with_tag_is_not_null_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_tag_is_not_null")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, tags={"env": "production", "region": "us"})
    _create_trace(store, trace2_id, exp_id, tags={"env": "staging"})
    _create_trace(store, trace3_id, exp_id, tags={})

    traces, _ = store.search_traces([exp_id], filter_string="tag.region IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}

    traces, _ = store.search_traces([exp_id], filter_string="tag.env IS NOT NULL")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='tag.region IS NOT NULL AND tag.env = "production"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id}


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_metadata_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_metadata_rlike")

    # Create traces with different metadata values
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(store, trace1_id, exp_id, trace_metadata={"version": "v1.2.3"})
    _create_trace(store, trace2_id, exp_id, trace_metadata={"version": "v2.0.0-beta"})
    _create_trace(store, trace3_id, exp_id, trace_metadata={"version": "v2.1.5"})

    # Test: RLIKE with semantic version pattern (no anchors for SQLite compatibility)
    traces, _ = store.search_traces(
        [exp_id], filter_string='metadata.version RLIKE "v[0-9]+\\.[0-9]+\\.[0-9]"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: RLIKE with version 2.x pattern
    traces, _ = store.search_traces([exp_id], filter_string='metadata.version RLIKE "v2\\.[0-9]"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: RLIKE matching beta versions
    traces, _ = store.search_traces([exp_id], filter_string='metadata.version RLIKE "beta"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_client_request_id_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_client_request_id_rlike")

    # Create traces with different client_request_id patterns
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id, client_request_id="req-prod-us-east-123")
    _create_trace(store, trace2_id, exp_id, client_request_id="req-prod-us-west-456")
    _create_trace(store, trace3_id, exp_id, client_request_id="req-staging-eu-789")
    _create_trace(store, trace4_id, exp_id, client_request_id="req-dev-local-001")

    # Test: RLIKE with pattern matching production requests
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "^req-prod"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with pattern matching US regions
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "us-(east|west)"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with digit pattern - all traces end with 3 digits
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "[0-9]{3}$"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id, trace4_id}

    # Test: RLIKE matching staging or dev environments
    traces, _ = store.search_traces(
        [exp_id], filter_string='trace.client_request_id RLIKE "(staging|dev)"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_span_type_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_type_rlike")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create spans with different types
    span1 = create_test_span_with_content(trace1_id, name="generate", span_id=111, span_type="LLM")
    span2 = create_test_span_with_content(
        trace2_id, name="embed", span_id=222, span_type="LLM_EMBEDDING"
    )
    span3 = create_test_span_with_content(
        trace3_id, name="retrieve", span_id=333, span_type="RETRIEVER"
    )
    span4 = create_test_span_with_content(
        trace4_id, name="chain", span_id=444, span_type="CHAIN_PARENT"
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])

    # Test: RLIKE with pattern matching LLM types (LLM or LLM_*)
    traces, _ = store.search_traces([exp_id], filter_string='span.type RLIKE "^LLM"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: RLIKE with pattern matching types ending with specific suffix
    traces, _ = store.search_traces([exp_id], filter_string='span.type RLIKE "PARENT$"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id

    # Test: RLIKE with character class for embedding or retriever
    traces, _ = store.search_traces(
        [exp_id], filter_string='span.type RLIKE "(EMBEDDING|RETRIEVER)"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}

    # Test: RLIKE matching underscore patterns
    traces, _ = store.search_traces([exp_id], filter_string='span.type RLIKE "_"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace4_id}


@pytest.mark.skipif(IS_MSSQL, reason="RLIKE is not supported for MSSQL database dialect.")
def test_search_traces_with_span_attributes_rlike_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_attributes_rlike")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    _create_trace(store, trace1_id, exp_id)
    _create_trace(store, trace2_id, exp_id)
    _create_trace(store, trace3_id, exp_id)
    _create_trace(store, trace4_id, exp_id)

    # Create spans with different custom attributes
    span1 = create_test_span_with_content(
        trace1_id,
        name="call1",
        span_id=111,
        span_type="LLM",
        custom_attributes={"model": "gpt-4-turbo-preview", "provider": "openai"},
    )
    span2 = create_test_span_with_content(
        trace2_id,
        name="call2",
        span_id=222,
        span_type="LLM",
        custom_attributes={"model": "gpt-3.5-turbo", "provider": "openai"},
    )
    span3 = create_test_span_with_content(
        trace3_id,
        name="call3",
        span_id=333,
        span_type="LLM",
        custom_attributes={"model": "claude-3-opus-20240229", "provider": "anthropic"},
    )
    span4 = create_test_span_with_content(
        trace4_id,
        name="call4",
        span_id=444,
        span_type="LLM",
        custom_attributes={"model": "claude-3-sonnet-20240229", "provider": "anthropic"},
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])
    store.log_spans(exp_id, [span3])
    store.log_spans(exp_id, [span4])

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model RLIKE "^gpt"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model RLIKE "^claude"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.model RLIKE "(preview|[0-9]{8})"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id, trace4_id}

    traces, _ = store.search_traces(
        [exp_id], filter_string='span.attributes.provider RLIKE "^openai"'
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string='span.attributes.model RLIKE "turbo"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}


def test_search_traces_with_empty_and_special_characters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_special_chars")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"

    _create_trace(
        store,
        trace1_id,
        exp_id,
        tags={"special": "test@example.com"},
        client_request_id="req-123",
    )
    _create_trace(
        store,
        trace2_id,
        exp_id,
        tags={"special": "user#admin"},
        client_request_id="req-456",
    )
    _create_trace(
        store,
        trace3_id,
        exp_id,
        tags={"special": "path/to/file"},
        client_request_id="req-789",
    )

    # Test: LIKE with @ character
    traces, _ = store.search_traces([exp_id], filter_string='tag.special LIKE "%@%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: LIKE with # character
    traces, _ = store.search_traces([exp_id], filter_string='tag.special LIKE "%#%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: LIKE with / character
    traces, _ = store.search_traces([exp_id], filter_string='tag.special LIKE "%/%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: ILIKE case-insensitive with special chars
    traces, _ = store.search_traces([exp_id], filter_string='tag.special ILIKE "%ADMIN%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


def test_search_traces_with_timestamp_ms_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_timestamp_ms")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    base_time = 1000000  # Use a fixed base time for consistency

    _create_trace(store, trace1_id, exp_id, request_time=base_time)
    _create_trace(store, trace2_id, exp_id, request_time=base_time + 5000)
    _create_trace(store, trace3_id, exp_id, request_time=base_time + 10000)
    _create_trace(store, trace4_id, exp_id, request_time=base_time + 15000)

    # Test: = (equals)
    traces, _ = store.search_traces([exp_id], filter_string=f"trace.timestamp_ms = {base_time}")
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: != (not equals)
    traces, _ = store.search_traces([exp_id], filter_string=f"trace.timestamp_ms != {base_time}")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}

    # Test: > (greater than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms > {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: >= (greater than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms >= {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}

    # Test: < (less than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms < {base_time + 10000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: <= (less than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.timestamp_ms <= {base_time + 10000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: Combined conditions (range query)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=f"trace.timestamp_ms >= {base_time + 5000} "
        f"AND trace.timestamp_ms < {base_time + 15000}",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}


def test_search_traces_with_execution_time_ms_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_execution_time_ms")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"
    trace5_id = "trace5"

    base_time = 1000000

    # Create traces with different execution times
    _create_trace(store, trace1_id, exp_id, request_time=base_time, execution_duration=100)
    _create_trace(store, trace2_id, exp_id, request_time=base_time, execution_duration=500)
    _create_trace(store, trace3_id, exp_id, request_time=base_time, execution_duration=1000)
    _create_trace(store, trace4_id, exp_id, request_time=base_time, execution_duration=2000)
    _create_trace(store, trace5_id, exp_id, request_time=base_time, execution_duration=5000)

    # Test: = (equals)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms = 1000")
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: != (not equals)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms != 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace4_id, trace5_id}

    # Test: > (greater than)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms > 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace4_id, trace5_id}

    # Test: >= (greater than or equal)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms >= 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id, trace5_id}

    # Test: < (less than)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms < 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: <= (less than or equal)
    traces, _ = store.search_traces([exp_id], filter_string="trace.execution_time_ms <= 1000")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: Combined conditions (find traces with execution time between 500ms and 2000ms)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string="trace.execution_time_ms >= 500 AND trace.execution_time_ms <= 2000",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}


def test_search_traces_with_end_time_ms_all_operators(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_end_time_ms_all_ops")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    base_time = 1000000

    # end_time_ms = timestamp_ms + execution_time_ms
    # trace1: starts at base_time, runs 1000ms -> ends at base_time + 1000
    # trace2: starts at base_time, runs 3000ms -> ends at base_time + 3000
    # trace3: starts at base_time, runs 5000ms -> ends at base_time + 5000
    # trace4: starts at base_time, runs 10000ms -> ends at base_time + 10000
    _create_trace(store, trace1_id, exp_id, request_time=base_time, execution_duration=1000)
    _create_trace(store, trace2_id, exp_id, request_time=base_time, execution_duration=3000)
    _create_trace(store, trace3_id, exp_id, request_time=base_time, execution_duration=5000)
    _create_trace(store, trace4_id, exp_id, request_time=base_time, execution_duration=10000)

    # Test: = (equals)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms = {base_time + 3000}"
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: != (not equals)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms != {base_time + 3000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace3_id, trace4_id}

    # Test: > (greater than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms > {base_time + 3000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: >= (greater than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms >= {base_time + 3000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id, trace4_id}

    # Test: < (less than)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms < {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: <= (less than or equal)
    traces, _ = store.search_traces(
        [exp_id], filter_string=f"trace.end_time_ms <= {base_time + 5000}"
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}

    # Test: Combined conditions (range query)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=f"trace.end_time_ms > {base_time + 1000} "
        f"AND trace.end_time_ms < {base_time + 10000}",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace2_id, trace3_id}


def test_search_traces_with_status_operators(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_status_operators")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    # Create traces with different statuses
    _create_trace(store, trace1_id, exp_id, state=TraceState.OK)
    _create_trace(store, trace2_id, exp_id, state=TraceState.OK)
    _create_trace(store, trace3_id, exp_id, state=TraceState.ERROR)
    _create_trace(store, trace4_id, exp_id, state=TraceState.IN_PROGRESS)

    # Test: = (equals) for OK status
    traces, _ = store.search_traces([exp_id], filter_string="trace.status = 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: = (equals) for ERROR status
    traces, _ = store.search_traces([exp_id], filter_string="trace.status = 'ERROR'")
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: != (not equals)
    traces, _ = store.search_traces([exp_id], filter_string="trace.status != 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: LIKE operator
    traces, _ = store.search_traces([exp_id], filter_string="trace.status LIKE 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: ILIKE operator
    traces, _ = store.search_traces([exp_id], filter_string="trace.status ILIKE 'error'")
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Using different aliases (attributes.status and status)
    traces, _ = store.search_traces([exp_id], filter_string="attributes.status = 'OK'")
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    traces, _ = store.search_traces([exp_id], filter_string="status = 'IN_PROGRESS'")
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


def test_search_traces_with_combined_numeric_and_string_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_combined_numeric_string")

    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    base_time = 1000000

    _create_trace(
        store,
        trace1_id,
        exp_id,
        request_time=base_time,
        execution_duration=100,
        tags={TraceTagKey.TRACE_NAME: "FastQuery"},
        state=TraceState.OK,
    )
    _create_trace(
        store,
        trace2_id,
        exp_id,
        request_time=base_time + 1000,
        execution_duration=500,
        tags={TraceTagKey.TRACE_NAME: "SlowQuery"},
        state=TraceState.OK,
    )
    _create_trace(
        store,
        trace3_id,
        exp_id,
        request_time=base_time + 2000,
        execution_duration=2000,
        tags={TraceTagKey.TRACE_NAME: "FastQuery"},
        state=TraceState.ERROR,
    )
    _create_trace(
        store,
        trace4_id,
        exp_id,
        request_time=base_time + 3000,
        execution_duration=5000,
        tags={TraceTagKey.TRACE_NAME: "SlowQuery"},
        state=TraceState.ERROR,
    )

    # Test: Fast queries (execution time < 1000ms) with OK status
    traces, _ = store.search_traces(
        [exp_id],
        filter_string="trace.execution_time_ms < 1000 AND trace.status = 'OK'",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: Slow queries (execution time >= 2000ms)
    traces, _ = store.search_traces(
        [exp_id],
        filter_string="trace.execution_time_ms >= 2000",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: Traces that started after base_time + 1000 with ERROR status
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=f"trace.timestamp_ms > {base_time + 1000} AND trace.status = 'ERROR'",
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace3_id, trace4_id}

    # Test: FastQuery traces with execution time < 500ms
    traces, _ = store.search_traces(
        [exp_id],
        filter_string='trace.name = "FastQuery" AND trace.execution_time_ms < 500',
    )
    assert len(traces) == 1
    assert traces[0].request_id == trace1_id

    # Test: Complex query with time range and name pattern
    traces, _ = store.search_traces(
        [exp_id],
        filter_string=(
            f"trace.timestamp_ms >= {base_time} "
            f"AND trace.timestamp_ms <= {base_time + 2000} "
            'AND trace.name LIKE "%Query%"'
        ),
    )
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id, trace3_id}


def test_search_traces_with_prompts_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_prompts_exact")

    # Create traces with different linked prompts
    trace1_id = "trace1"
    trace2_id = "trace2"
    trace3_id = "trace3"
    trace4_id = "trace4"

    # Trace 1: linked to qa-agent-system-prompt version 4
    _create_trace(store, trace1_id, exp_id)
    store.link_prompts_to_trace(
        trace1_id, [PromptVersion(name="qa-agent-system-prompt", version=4, template="")]
    )

    # Trace 2: linked to qa-agent-system-prompt version 5
    _create_trace(store, trace2_id, exp_id)
    store.link_prompts_to_trace(
        trace2_id, [PromptVersion(name="qa-agent-system-prompt", version=5, template="")]
    )

    # Trace 3: linked to chat-assistant-prompt version 1
    _create_trace(store, trace3_id, exp_id)
    store.link_prompts_to_trace(
        trace3_id, [PromptVersion(name="chat-assistant-prompt", version=1, template="")]
    )

    # Trace 4: linked to multiple prompts
    _create_trace(store, trace4_id, exp_id)
    store.link_prompts_to_trace(
        trace4_id,
        [
            PromptVersion(name="qa-agent-system-prompt", version=4, template=""),
            PromptVersion(name="chat-assistant-prompt", version=2, template=""),
        ],
    )

    # Test: Filter by exact prompt name/version
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "qa-agent-system-prompt/4"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace4_id}

    # Test: Filter by another exact prompt name/version
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "qa-agent-system-prompt/5"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Filter by chat assistant prompt
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "chat-assistant-prompt/1"')
    assert len(traces) == 1
    assert traces[0].request_id == trace3_id

    # Test: Filter by prompt that appears in multiple trace
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "chat-assistant-prompt/2"')
    assert len(traces) == 1
    assert traces[0].request_id == trace4_id


@pytest.mark.parametrize(
    ("comparator", "filter_string"),
    [
        ("LIKE", 'prompt LIKE "%qa-agent%"'),
        ("ILIKE", 'prompt ILIKE "%CHAT%"'),
        ("RLIKE", 'prompt RLIKE "version.*1"'),
        ("!=", 'prompt != "test/1"'),
    ],
)
def test_search_traces_with_prompts_filter_invalid_comparator(
    store: SqlAlchemyStore, comparator: str, filter_string: str
):
    exp_id = store.create_experiment("test_prompts_invalid")

    with pytest.raises(
        MlflowException,
        match=f"Invalid comparator '{comparator}' for prompts filter. "
        "Only '=' is supported with format: prompt = \"name/version\"",
    ):
        store.search_traces([exp_id], filter_string=filter_string)


@pytest.mark.parametrize(
    ("filter_string", "invalid_value"),
    [
        ('prompt = "qa-agent-system-prompt"', "qa-agent-system-prompt"),
        ('prompt = "foo/1/baz"', "foo/1/baz"),
        ('prompt = ""', ""),
    ],
)
def test_search_traces_with_prompts_filter_invalid_format(
    store: SqlAlchemyStore, filter_string: str, invalid_value: str
):
    exp_id = store.create_experiment("test_prompts_invalid_format")

    with pytest.raises(
        MlflowException,
        match=f"Invalid prompts filter value '{invalid_value}'. "
        'Expected format: prompt = "name/version"',
    ):
        store.search_traces([exp_id], filter_string=filter_string)


def test_search_traces_with_prompts_filter_no_matches(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_prompts_no_match")

    # Create traces with linked prompts
    trace1_id = "trace1"
    _create_trace(store, trace1_id, exp_id)
    store.link_prompts_to_trace(
        trace1_id, [PromptVersion(name="qa-agent-system-prompt", version=4, template="")]
    )

    # Test: Filter by non-existent prompt
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "non-existent-prompt/999"')
    assert len(traces) == 0

    # Test: Filter by correct name but wrong version
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "qa-agent-system-prompt/999"')
    assert len(traces) == 0


def test_search_traces_with_prompts_filter_multiple_prompts(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_prompts_multiple")

    # Create traces with multiple linked prompts
    trace1_id = "trace1"
    trace2_id = "trace2"

    # Trace 1: Single prompt
    _create_trace(store, trace1_id, exp_id)
    store.link_prompts_to_trace(trace1_id, [PromptVersion(name="prompt-a", version=1, template="")])

    # Trace 2: Multiple prompts
    _create_trace(store, trace2_id, exp_id)
    store.link_prompts_to_trace(
        trace2_id,
        [
            PromptVersion(name="prompt-a", version=1, template=""),
            PromptVersion(name="prompt-b", version=2, template=""),
            PromptVersion(name="prompt-c", version=3, template=""),
        ],
    )

    # Test: Filter by first prompt - should match both
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "prompt-a/1"')
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1_id, trace2_id}

    # Test: Filter by second prompt - should only match trace2
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "prompt-b/2"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id

    # Test: Filter by third prompt - should only match trace2
    traces, _ = store.search_traces([exp_id], filter_string='prompt = "prompt-c/3"')
    assert len(traces) == 1
    assert traces[0].request_id == trace2_id


def test_search_traces_with_span_attributute_backticks(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_attribute_backticks")
    trace_info_1 = _create_trace(store, "trace_1", exp_id)
    trace_info_2 = _create_trace(store, "trace_2", exp_id)

    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace1",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_info_1.trace_id, cls=TraceJSONEncoder),
                "mlflow.experimentId": json.dumps(exp_id, cls=TraceJSONEncoder),
                "mlflow.spanInputs": json.dumps({"input": "test1"}, cls=TraceJSONEncoder),
            },
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        trace_info_1.trace_id,
        "LLM",
    )

    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace2",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_info_2.trace_id, cls=TraceJSONEncoder),
                "mlflow.experimentId": json.dumps(exp_id, cls=TraceJSONEncoder),
                "mlflow.spanInputs": json.dumps({"input": "test2"}, cls=TraceJSONEncoder),
            },
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        trace_info_2.trace_id,
        "LLM",
    )

    store.log_spans(exp_id, [span1])
    store.log_spans(exp_id, [span2])

    traces, _ = store.search_traces([exp_id], filter_string='trace.text ILIKE "%test1%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace_info_1.trace_id

    traces, _ = store.search_traces([exp_id], filter_string='trace.text ILIKE "%test2%"')
    assert len(traces) == 1
    assert traces[0].request_id == trace_info_2.trace_id


def test_set_and_delete_tags(store: SqlAlchemyStore):
    exp1 = store.create_experiment("exp1")
    trace_id = "tr-123"
    _create_trace(store, trace_id, experiment_id=exp1)

    # Delete system tag for easier testing
    store.delete_trace_tag(trace_id, MLFLOW_ARTIFACT_LOCATION)

    assert store.get_trace_info(trace_id).tags == {}

    store.set_trace_tag(trace_id, "tag1", "apple")
    assert store.get_trace_info(trace_id).tags == {"tag1": "apple"}

    store.set_trace_tag(trace_id, "tag1", "grape")
    assert store.get_trace_info(trace_id).tags == {"tag1": "grape"}

    store.set_trace_tag(trace_id, "tag2", "orange")
    assert store.get_trace_info(trace_id).tags == {"tag1": "grape", "tag2": "orange"}

    store.delete_trace_tag(trace_id, "tag1")
    assert store.get_trace_info(trace_id).tags == {"tag2": "orange"}

    # test value length
    store.set_trace_tag(trace_id, "key", "v" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE)
    assert store.get_trace_info(trace_id).tags["key"] == "v" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE

    with pytest.raises(MlflowException, match="No trace tag with key 'tag1'"):
        store.delete_trace_tag(trace_id, "tag1")


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        (None, "value", "Missing value for required parameter 'key'"),
        (
            "invalid?tag!name:(",
            "value",
            "Invalid value \"invalid\\?tag!name:\\(\" for parameter 'key' supplied",
        ),
        (
            "/.:\\.",
            "value",
            "Invalid value \"/\\.:\\\\\\\\.\" for parameter 'key' supplied",
        ),
        ("../", "value", "Invalid value \"\\.\\./\" for parameter 'key' supplied"),
        ("a" * 251, "value", "'key' exceeds the maximum length of 250 characters"),
    ],
    # Name each test case too avoid including the long string arguments in the test name
    ids=["null-key", "bad-key-1", "bad-key-2", "bad-key-3", "too-long-key"],
)
def test_set_invalid_tag(key, value, expected_error, store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match=expected_error):
        store.set_trace_tag("tr-123", key, value)


def test_set_tag_truncate_too_long_tag(store: SqlAlchemyStore):
    exp1 = store.create_experiment("exp1")
    trace_id = "tr-123"
    _create_trace(store, trace_id, experiment_id=exp1)

    store.set_trace_tag(trace_id, "key", "123" + "a" * 8000)
    tags = store.get_trace_info(trace_id).tags
    assert len(tags["key"]) == 8000
    assert tags["key"] == "123" + "a" * 7997


def test_delete_traces(store):
    exp1 = store.create_experiment("exp1")
    exp2 = store.create_experiment("exp2")
    now = int(time.time() * 1000)

    for i in range(10):
        _create_trace(
            store,
            f"tr-exp1-{i}",
            exp1,
            tags={"tag": "apple"},
            trace_metadata={"rq": "foo"},
        )
        _create_trace(
            store,
            f"tr-exp2-{i}",
            exp2,
            tags={"tag": "orange"},
            trace_metadata={"rq": "bar"},
        )

    traces, _ = store.search_traces([exp1, exp2])
    assert len(traces) == 20

    deleted = store.delete_traces(experiment_id=exp1, max_timestamp_millis=now)
    assert deleted == 10
    traces, _ = store.search_traces([exp1, exp2])
    assert len(traces) == 10
    for trace in traces:
        assert trace.experiment_id == exp2

    deleted = store.delete_traces(experiment_id=exp2, max_timestamp_millis=now)
    assert deleted == 10
    traces, _ = store.search_traces([exp1, exp2])
    assert len(traces) == 0

    deleted = store.delete_traces(experiment_id=exp1, max_timestamp_millis=now)
    assert deleted == 0


def test_delete_traces_with_max_timestamp(store):
    exp1 = store.create_experiment("exp1")
    for i in range(10):
        _create_trace(store, f"tr-{i}", exp1, request_time=i)

    deleted = store.delete_traces(exp1, max_timestamp_millis=3)
    assert deleted == 4  # inclusive (0, 1, 2, 3)
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 6
    for trace in traces:
        assert trace.timestamp_ms >= 4

    deleted = store.delete_traces(exp1, max_timestamp_millis=10)
    assert deleted == 6
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 0


def test_delete_traces_with_max_count(store):
    exp1 = store.create_experiment("exp1")
    for i in range(10):
        _create_trace(store, f"tr-{i}", exp1, request_time=i)

    deleted = store.delete_traces(exp1, max_traces=4, max_timestamp_millis=10)
    assert deleted == 4
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 6
    # Traces should be deleted from the oldest
    for trace in traces:
        assert trace.timestamp_ms >= 4

    deleted = store.delete_traces(exp1, max_traces=10, max_timestamp_millis=8)
    assert deleted == 5
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 1


def test_delete_traces_with_trace_ids(store):
    exp1 = store.create_experiment("exp1")
    for i in range(10):
        _create_trace(store, f"tr-{i}", exp1, request_time=i)

    deleted = store.delete_traces(exp1, trace_ids=[f"tr-{i}" for i in range(8)])
    assert deleted == 8
    traces, _ = store.search_traces([exp1])
    assert len(traces) == 2
    assert [trace.trace_id for trace in traces] == ["tr-9", "tr-8"]


def test_delete_traces_raises_error(store):
    exp_id = store.create_experiment("test")

    with pytest.raises(
        MlflowException,
        match=r"Either `max_timestamp_millis` or `trace_ids` must be specified.",
    ):
        store.delete_traces(exp_id)
    with pytest.raises(
        MlflowException,
        match=r"Only one of `max_timestamp_millis` and `trace_ids` can be specified.",
    ):
        store.delete_traces(exp_id, max_timestamp_millis=100, trace_ids=["trace_id"])
    with pytest.raises(
        MlflowException,
        match=r"`max_traces` can't be specified if `trace_ids` is specified.",
    ):
        store.delete_traces(exp_id, max_traces=2, trace_ids=["trace_id"])
    with pytest.raises(
        MlflowException, match=r"`max_traces` must be a positive integer, received 0"
    ):
        store.delete_traces(exp_id, 100, max_traces=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment and trace first
    experiment_id = store.create_experiment("test_span_experiment")
    trace_info = TraceInfo(
        trace_id="tr-span-test-123",
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.OK,
    )
    trace_info = store.start_trace(trace_info)

    # Create a mock OpenTelemetry span

    # Create mock context
    mock_context = mock.Mock()
    mock_context.trace_id = 12345
    mock_context.span_id = 222 if not is_async else 333
    mock_context.is_remote = False
    mock_context.trace_flags = trace_api.TraceFlags(1)
    mock_context.trace_state = trace_api.TraceState()  # Empty TraceState

    parent_mock_context = mock.Mock()
    parent_mock_context.trace_id = 12345
    parent_mock_context.span_id = 111
    parent_mock_context.is_remote = False
    parent_mock_context.trace_flags = trace_api.TraceFlags(1)
    parent_mock_context.trace_state = trace_api.TraceState()  # Empty TraceState

    readable_span = OTelReadableSpan(
        name="test_span",
        context=mock_context,
        parent=parent_mock_context if not is_async else None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_info.trace_id),
            "mlflow.spanInputs": json.dumps({"input": "test_input"}, cls=TraceJSONEncoder),
            "mlflow.spanOutputs": json.dumps({"output": "test_output"}, cls=TraceJSONEncoder),
            "mlflow.spanType": json.dumps("LLM" if not is_async else "CHAIN", cls=TraceJSONEncoder),
            "custom_attr": json.dumps("custom_value", cls=TraceJSONEncoder),
        },
        start_time=1000000000 if not is_async else 3000000000,
        end_time=2000000000 if not is_async else 4000000000,
        resource=_OTelResource.get_empty(),
    )

    # Create MLflow span from OpenTelemetry span
    span = create_mlflow_span(readable_span, trace_info.trace_id, "LLM")
    assert isinstance(span, Span)

    # Test logging the span using sync or async method
    if is_async:
        logged_spans = await store.log_spans_async(experiment_id, [span])
    else:
        logged_spans = store.log_spans(experiment_id, [span])

    # Verify the returned spans are the same
    assert len(logged_spans) == 1
    assert logged_spans[0] == span
    assert logged_spans[0].trace_id == trace_info.trace_id
    assert logged_spans[0].span_id == span.span_id

    # Verify the span was saved to the database
    with store.ManagedSessionMaker() as session:
        saved_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_info.trace_id, SqlSpan.span_id == span.span_id)
            .first()
        )

        assert saved_span is not None
        assert saved_span.experiment_id == int(experiment_id)
        assert saved_span.parent_span_id == span.parent_id
        assert saved_span.status == "UNSET"  # Default OpenTelemetry status
        assert saved_span.status == span.status.status_code
        assert saved_span.start_time_unix_nano == span.start_time_ns
        assert saved_span.end_time_unix_nano == span.end_time_ns
        # Check the computed duration
        assert saved_span.duration_ns == (span.end_time_ns - span.start_time_ns)

        # Verify the content is properly serialized
        content_dict = json.loads(saved_span.content)
        assert content_dict["name"] == "test_span"
        # Inputs and outputs are stored in attributes as strings
        assert content_dict["attributes"]["mlflow.spanInputs"] == json.dumps(
            {"input": "test_input"}, cls=TraceJSONEncoder
        )
        assert content_dict["attributes"]["mlflow.spanOutputs"] == json.dumps(
            {"output": "test_output"}, cls=TraceJSONEncoder
        )
        expected_type = "LLM" if not is_async else "CHAIN"
        assert content_dict["attributes"]["mlflow.spanType"] == json.dumps(
            expected_type, cls=TraceJSONEncoder
        )


def test_log_spans_multiple_traces(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_multi_trace_experiment")

    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace1",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps("tr-multi-1", cls=TraceJSONEncoder)},
            start_time=1000000000,
            end_time=2000000000,
            resource=_OTelResource.get_empty(),
        ),
        "tr-multi-1",
    )

    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="span_trace2",
            context=trace_api.SpanContext(
                trace_id=67890,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps("tr-multi-2", cls=TraceJSONEncoder)},
            start_time=3000000000,
            end_time=4000000000,
            resource=_OTelResource.get_empty(),
        ),
        "tr-multi-2",
    )

    # Multi-trace log_spans should succeed in a single call
    result = store.log_spans(experiment_id, [span1, span2])
    assert len(result) == 2

    # Verify both traces were created with correct spans in the database
    with store.ManagedSessionMaker() as session:
        trace1 = session.query(SqlTraceInfo).filter_by(request_id="tr-multi-1").one()
        assert trace1.experiment_id == int(experiment_id)

        trace2 = session.query(SqlTraceInfo).filter_by(request_id="tr-multi-2").one()
        assert trace2.experiment_id == int(experiment_id)

        span_row1 = session.query(SqlSpan).filter_by(trace_id="tr-multi-1").one()
        assert span_row1.name == "span_trace1"

        span_row2 = session.query(SqlSpan).filter_by(trace_id="tr-multi-2").one()
        assert span_row2.name == "span_trace2"


def test_log_spans_persists_links(store: SqlAlchemyStore):
    trace_id = "tr-links-test"
    experiment_id = store.create_experiment("test_links_experiment")

    span = create_test_span(
        trace_id=trace_id,
        links=[
            Link(trace_id="tr-abc123", span_id="aabbccddeeff0011", attributes={"type": "causal"}),
            Link(trace_id="tr-def456", span_id="1122334455667788"),
        ],
    )

    store.log_spans(experiment_id, [span])

    # Verify links survive the full round-trip via get_trace
    trace = store.get_trace(trace_id)
    retrieved_span = trace.data.spans[0]
    assert len(retrieved_span.links) == 2
    assert retrieved_span.links[0].trace_id == "tr-abc123"
    assert retrieved_span.links[0].span_id == "aabbccddeeff0011"
    assert retrieved_span.links[0].attributes == {"type": "causal"}
    assert retrieved_span.links[1].trace_id == "tr-def456"
    assert retrieved_span.links[1].span_id == "1122334455667788"
    assert retrieved_span.links[1].attributes is None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_creates_trace_if_not_exists(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment but no trace
    experiment_id = store.create_experiment("test_auto_trace_experiment")

    # Create a span without a pre-existing trace
    trace_id = "tr-auto-created-trace"
    readable_span = OTelReadableSpan(
        name="auto_trace_span",
        context=trace_api.SpanContext(
            trace_id=98765,
            span_id=555,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
            "mlflow.experimentId": json.dumps(experiment_id, cls=TraceJSONEncoder),
        },
        start_time=5000000000,
        end_time=6000000000,
        resource=_OTelResource.get_empty(),
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Log the span - should create the trace automatically
    if is_async:
        logged_spans = await store.log_spans_async(experiment_id, [span])
    else:
        logged_spans = store.log_spans(experiment_id, [span])

    assert len(logged_spans) == 1
    assert logged_spans[0] == span

    # Verify the trace was created
    with store.ManagedSessionMaker() as session:
        created_trace = (
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).first()
        )

        assert created_trace is not None
        assert created_trace.experiment_id == int(experiment_id)
        assert created_trace.timestamp_ms == 5000000000 // 1_000_000
        assert created_trace.execution_time_ms == 1000000000 // 1_000_000
        # When root span status is UNSET (unexpected), we assume trace status is OK
        assert created_trace.status == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_empty_list(store: SqlAlchemyStore, is_async: bool):
    experiment_id = store.create_experiment("test_empty_experiment")

    if is_async:
        result = await store.log_spans_async(experiment_id, [])
    else:
        result = store.log_spans(experiment_id, [])
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_concurrent_trace_creation(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment
    experiment_id = store.create_experiment("test_concurrent_trace")
    trace_id = "tr-concurrent-test"

    # Create a span
    readable_span = OTelReadableSpan(
        name="concurrent_span",
        context=trace_api.SpanContext(
            trace_id=12345,
            span_id=999,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        resource=_OTelResource.get_empty(),
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        },
        start_time=1000000000,
        end_time=2000000000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        events=[],
        links=[],
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Simulate a race condition where flush() raises IntegrityError
    # This tests that the code properly handles concurrent trace creation
    original_flush = None
    call_count = 0

    def mock_flush(self):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call to flush (for trace creation) raises IntegrityError
            raise IntegrityError("UNIQUE constraint failed", None, None)
        else:
            # Subsequent calls work normally
            return original_flush()

    with store.ManagedSessionMaker() as session:
        original_flush = session.flush
        with mock.patch.object(session, "flush", mock_flush):
            # This should handle the IntegrityError and still succeed
            if is_async:
                result = await store.log_spans_async(experiment_id, [span])
            else:
                result = store.log_spans(experiment_id, [span])

    # Verify the span was logged successfully despite the race condition
    assert len(result) == 1
    assert result[0] == span

    # Verify the trace and span exist in the database
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.experiment_id == int(experiment_id)

        saved_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_id, SqlSpan.span_id == span.span_id)
            .one()
        )
        assert saved_span is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_trace_time_range(store: SqlAlchemyStore, is_async: bool):
    experiment_id = _create_experiments(store, "test_log_spans_updates_trace")
    trace_id = "tr-time-update-test-123"

    # Create first span from 1s to 2s
    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="early_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=1_000_000_000,  # 1 second in nanoseconds
            end_time=2_000_000_000,  # 2 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log first span - creates trace with 1s start, 1s duration
    if is_async:
        await store.log_spans_async(experiment_id, [span1])
    else:
        store.log_spans(experiment_id, [span1])

    # Verify initial trace times
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 1_000  # 1 second
        assert trace.execution_time_ms == 1_000  # 1 second duration

    # Create second span that starts earlier (0.5s) and ends later (3s)
    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="extended_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=500_000_000,  # 0.5 seconds
            end_time=3_000_000_000,  # 3 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log second span - should update trace to 0.5s start, 2.5s duration
    if is_async:
        await store.log_spans_async(experiment_id, [span2])
    else:
        store.log_spans(experiment_id, [span2])

    # Verify trace times were updated
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # 0.5 seconds (earlier start)
        assert trace.execution_time_ms == 2_500  # 2.5 seconds duration (0.5s to 3s)

    # Create third span that only extends the end time (2.5s to 4s)
    span3 = create_mlflow_span(
        OTelReadableSpan(
            name="later_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=333,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=2_500_000_000,  # 2.5 seconds
            end_time=4_000_000_000,  # 4 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log third span - should only update end time
    if is_async:
        await store.log_spans_async(experiment_id, [span3])
    else:
        store.log_spans(experiment_id, [span3])

    # Verify trace times were updated again
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Still 0.5 seconds (no earlier start)
        assert trace.execution_time_ms == 3_500  # 3.5 seconds duration (0.5s to 4s)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_no_end_time(store: SqlAlchemyStore, is_async: bool):
    experiment_id = _create_experiments(store, "test_log_spans_no_end_time")
    trace_id = "tr-no-end-time-test-123"

    # Create span without end time (in-progress span)
    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="in_progress_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=1_000_000_000,  # 1 second in nanoseconds
            end_time=None,  # No end time - span still in progress
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log span with no end time
    if is_async:
        await store.log_spans_async(experiment_id, [span1])
    else:
        store.log_spans(experiment_id, [span1])

    # Verify trace has timestamp but no execution_time
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 1_000  # 1 second
        assert trace.execution_time_ms is None  # No execution time since span not ended

    # Add a second span that also has no end time
    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="another_in_progress_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=500_000_000,  # 0.5 seconds - earlier start
            end_time=None,  # No end time
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log second span with no end time
    if is_async:
        await store.log_spans_async(experiment_id, [span2])
    else:
        store.log_spans(experiment_id, [span2])

    # Verify trace timestamp updated but execution_time still None
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Updated to earlier time
        assert trace.execution_time_ms is None  # Still no execution time

    # Now add a span with an end time
    span3 = create_mlflow_span(
        OTelReadableSpan(
            name="completed_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=333,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=2_000_000_000,  # 2 seconds
            end_time=3_000_000_000,  # 3 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log span with end time
    if is_async:
        await store.log_spans_async(experiment_id, [span3])
    else:
        store.log_spans(experiment_id, [span3])

    # Verify trace now has execution_time
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Still earliest start
        assert trace.execution_time_ms == 2_500  # 3s - 0.5s = 2.5s


def test_batch_get_traces_basic(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_batch_get_traces")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="root_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="child_span",
            span_id=222,
            parent_id=111,
            status=trace_api.StatusCode.UNSET,
            start_ns=1_500_000_000,
            end_ns=1_800_000_000,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 2

    root_span = next(s for s in loaded_spans if s.name == "root_span")
    child_span = next(s for s in loaded_spans if s.name == "child_span")

    assert root_span.trace_id == trace_id
    assert root_span.span_id == "000000000000006f"
    assert root_span.parent_id is None
    assert root_span.start_time_ns == 1_000_000_000
    assert root_span.end_time_ns == 2_000_000_000

    assert child_span.trace_id == trace_id
    assert child_span.span_id == "00000000000000de"
    assert child_span.parent_id == "000000000000006f"
    assert child_span.start_time_ns == 1_500_000_000
    assert child_span.end_time_ns == 1_800_000_000


def test_batch_get_traces_empty_trace(store: SqlAlchemyStore) -> None:
    trace_id = f"tr-{uuid.uuid4().hex}"
    traces = store.batch_get_traces([trace_id])
    assert traces == []


def test_batch_get_traces_ordering(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_ordering")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="second_span",
            span_id=222,
            start_ns=2_000_000_000,
            end_ns=3_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="first_span",
            span_id=111,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="third_span",
            span_id=333,
            start_ns=3_000_000_000,
            end_ns=4_000_000_000,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 3
    assert loaded_spans[0].name == "first_span"
    assert loaded_spans[1].name == "second_span"
    assert loaded_spans[2].name == "third_span"

    assert loaded_spans[0].start_time_ns == 1_000_000_000
    assert loaded_spans[1].start_time_ns == 2_000_000_000
    assert loaded_spans[2].start_time_ns == 3_000_000_000


def test_batch_get_traces_with_complex_attributes(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_complex")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="complex_span",
        status_code=trace_api.StatusCode.ERROR,
        status_description="Test error",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )

    otel_span._attributes = {
        "llm.model_name": "gpt-4",
        "llm.input_tokens": 100,
        "llm.output_tokens": 50,
        "custom.key": "custom_value",
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
    }

    span = create_mlflow_span(otel_span, trace_id, "LLM")

    store.log_spans(experiment_id, [span])
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 1
    loaded_span = loaded_spans[0]

    assert loaded_span.status.status_code == "ERROR"
    assert loaded_span.status.description == "Test error"

    assert loaded_span.attributes.get("llm.model_name") == "gpt-4"
    assert loaded_span.attributes.get("llm.input_tokens") == 100
    assert loaded_span.attributes.get("llm.output_tokens") == 50
    assert loaded_span.attributes.get("custom.key") == "custom_value"


def test_batch_get_traces_multiple_traces(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_multiple")
    trace_id_1 = f"tr-{uuid.uuid4().hex}"
    trace_id_2 = f"tr-{uuid.uuid4().hex}"

    spans_trace_1 = [
        create_test_span(
            trace_id=trace_id_1,
            name="trace1_span1",
            span_id=111,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id_1,
            name="trace1_span2",
            span_id=112,
            trace_num=12345,
        ),
    ]

    spans_trace_2 = [
        create_test_span(
            trace_id=trace_id_2,
            name="trace2_span1",
            span_id=221,
            trace_num=67890,
        ),
    ]

    store.log_spans(experiment_id, spans_trace_1)
    store.log_spans(experiment_id, spans_trace_2)
    traces = store.batch_get_traces([trace_id_1, trace_id_2])

    assert len(traces) == 2

    # Find traces by ID since order might not be guaranteed
    trace_1 = next(t for t in traces if t.info.trace_id == trace_id_1)
    trace_2 = next(t for t in traces if t.info.trace_id == trace_id_2)

    loaded_spans_1 = trace_1.data.spans
    loaded_spans_2 = trace_2.data.spans

    assert len(loaded_spans_1) == 2
    assert len(loaded_spans_2) == 1

    trace_1_spans = [span.to_dict() for span in loaded_spans_1]
    trace_2_spans = [span.to_dict() for span in loaded_spans_2]

    assert [span.to_dict() for span in loaded_spans_1] == trace_1_spans
    assert [span.to_dict() for span in loaded_spans_2] == trace_2_spans


def test_batch_get_traces_preserves_json_serialization(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_load_spans_json")
    trace_id = f"tr-{uuid.uuid4().hex}"

    original_span = create_test_span(
        trace_id=trace_id,
        name="json_test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )

    store.log_spans(experiment_id, [original_span])
    traces = store.batch_get_traces([trace_id])

    assert len(traces) == 1
    loaded_spans = traces[0].data.spans

    assert len(loaded_spans) == 1
    loaded_span = loaded_spans[0]

    assert loaded_span.name == original_span.name
    assert loaded_span.trace_id == original_span.trace_id
    assert loaded_span.span_id == original_span.span_id
    assert loaded_span.start_time_ns == original_span.start_time_ns
    assert loaded_span.end_time_ns == original_span.end_time_ns


def test_batch_get_traces_integration_with_trace_handler(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_integration")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="integration_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACKING_STORE.value

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    loaded_spans = traces[0].data.spans
    assert len(loaded_spans) == 1
    assert loaded_spans[0].name == "integration_span"


def test_batch_get_traces_with_incomplete_trace(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_incomplete_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="incomplete_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.SIZE_STATS: json.dumps({
                    TraceSizeStatsKey.NUM_SPANS: 2,
                }),
            },
        )
    )
    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 0

    # add another complete trace
    trace_id_2 = f"tr-{uuid.uuid4().hex}"
    spans = [
        create_test_span(
            trace_id=trace_id_2,
            name="incomplete_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]
    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id_2,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
        )
    )
    traces = store.batch_get_traces([trace_id, trace_id_2])
    assert len(traces) == 1
    assert traces[0].info.trace_id == trace_id_2
    assert traces[0].info.status == TraceState.OK
    assert len(traces[0].data.spans) == 1
    assert traces[0].data.spans[0].name == "incomplete_span"
    assert traces[0].data.spans[0].status.status_code == "OK"


def test_batch_get_traces_raises_for_artifact_repo_traces(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_artifact_repo_traces")

    # Create a trace via start_trace only (no log_spans call),
    # so it has no SPANS_LOCATION tag — simulating spans stored in artifact repo.
    artifact_trace_id = f"tr-{uuid.uuid4().hex}"
    store.start_trace(
        TraceInfo(
            trace_id=artifact_trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1000,
            execution_duration=500,
            state=TraceState.OK,
        )
    )

    with pytest.raises(MlflowTracingException, match="not stored in tracking store"):
        store.batch_get_traces([artifact_trace_id])


def test_log_spans_token_usage(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_token_usage")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )

    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }

    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    # verify token usage is stored in the trace info
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.token_usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }

    # verify loaded trace has same token usage
    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage is not None
    assert trace.info.token_usage["input_tokens"] == 100
    assert trace.info.token_usage["output_tokens"] == 50
    assert trace.info.token_usage["total_tokens"] == 150


def test_log_spans_update_token_usage_incrementally(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_update_token_usage")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage["input_tokens"] == 100
    assert trace.info.token_usage["output_tokens"] == 50
    assert trace.info.token_usage["total_tokens"] == 150

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 200,
            "output_tokens": 75,
            "total_tokens": 275,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.token_usage["input_tokens"] == 300
    assert trace.info.token_usage["output_tokens"] == 125
    assert trace.info.token_usage["total_tokens"] == 425


def test_log_spans_cost(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_cost")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )

    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
        }),
    }

    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    # verify cost is stored in the trace info
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.cost == {
        "input_cost": 0.01,
        "output_cost": 0.02,
        "total_cost": 0.03,
    }

    # verify loaded trace has same cost
    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.cost is not None
    assert trace.info.cost["input_cost"] == 0.01
    assert trace.info.cost["output_cost"] == 0.02
    assert trace.info.cost["total_cost"] == 0.03


def test_log_spans_update_cost_incrementally(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_update_cost")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.cost["input_cost"] == 0.01
    assert trace.info.cost["output_cost"] == 0.02
    assert trace.info.cost["total_cost"] == 0.03

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 0.005,
            "output_cost": 0.01,
            "total_cost": 0.015,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    traces = store.batch_get_traces([trace_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.cost["input_cost"] == 0.015
    assert trace.info.cost["output_cost"] == 0.03
    assert trace.info.cost["total_cost"] == 0.045


def test_log_spans_does_not_overwrite_finalized_trace_info(store: SqlAlchemyStore) -> None:
    """start_trace() sets TRACE_INFO_FINALIZED; subsequent log_spans() must not overwrite
    request_time, execution_duration, session_id, token_usage, or cost.
    """
    experiment_id = store.create_experiment("test_trace_info_finalized")
    trace_id = f"tr-{uuid.uuid4().hex}"

    # start_trace() writes authoritative trace-level values and sets TRACE_INFO_FINALIZED.
    authoritative_request_time = 1_000
    authoritative_duration = 500
    authoritative_session = "session-from-start-trace"
    authoritative_token_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    authoritative_cost = {"input_cost": 0.001, "output_cost": 0.0005, "total_cost": 0.0015}

    _create_trace(
        store,
        trace_id,
        experiment_id,
        request_time=authoritative_request_time,
        execution_duration=authoritative_duration,
        trace_metadata={
            TraceMetadataKey.TRACE_SESSION: authoritative_session,
            TraceMetadataKey.TOKEN_USAGE: json.dumps(authoritative_token_usage),
            TraceMetadataKey.COST: json.dumps(authoritative_cost),
        },
    )

    # log_spans() arrives with different values that should all be ignored.
    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000,  # earlier start (ms=1) — should NOT update request_time
        end_time=9_000_000_000,  # later end — should NOT update execution_duration
        trace_id_num=99999,
        span_id_num=1,
    )
    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        "session.id": "session-from-log-spans",
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 999,
            "output_tokens": 999,
            "total_tokens": 1998,
        }),
        SpanAttributeKey.LLM_COST: json.dumps({
            "input_cost": 9.99,
            "output_cost": 9.99,
            "total_cost": 19.98,
        }),
    }
    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    trace_info = store.get_trace_info(trace_id)

    # Timestamp and duration unchanged
    assert trace_info.request_time == authoritative_request_time
    assert trace_info.execution_duration == authoritative_duration

    # Session ID unchanged
    assert trace_info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == authoritative_session

    # Token usage unchanged
    assert trace_info.token_usage == authoritative_token_usage

    # Cost unchanged
    stored_cost = json.loads(trace_info.trace_metadata[TraceMetadataKey.COST])
    assert stored_cost == authoritative_cost


def test_batch_get_traces_token_usage(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_batch_get_traces_token_usage")

    trace_id_1 = f"tr-{uuid.uuid4().hex}"
    otel_span1 = create_test_otel_span(
        trace_id=trace_id_1,
        name="trace1_span",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id_1, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id_1, "LLM")
    store.log_spans(experiment_id, [span1])

    trace_id_2 = f"tr-{uuid.uuid4().hex}"
    otel_span2 = create_test_otel_span(
        trace_id=trace_id_2,
        name="trace2_span",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=67890,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id_2, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id_2, "LLM")
    store.log_spans(experiment_id, [span2])

    trace_id_3 = f"tr-{uuid.uuid4().hex}"
    otel_span3 = create_test_otel_span(
        trace_id=trace_id_3,
        name="trace3_span",
        start_time=5_000_000_000,
        end_time=6_000_000_000,
        trace_id_num=11111,
        span_id_num=333,
    )
    otel_span3._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id_3, cls=TraceJSONEncoder),
    }
    span3 = create_mlflow_span(otel_span3, trace_id_3, "UNKNOWN")
    store.log_spans(experiment_id, [span3])

    trace_infos = [
        store.get_trace_info(trace_id) for trace_id in [trace_id_1, trace_id_2, trace_id_3]
    ]
    assert trace_infos[0].token_usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    assert trace_infos[1].token_usage == {
        "input_tokens": 200,
        "output_tokens": 100,
        "total_tokens": 300,
    }
    assert trace_infos[2].token_usage is None

    traces = store.batch_get_traces([trace_id_1, trace_id_2, trace_id_3])
    assert len(traces) == 3

    traces_by_id = {trace.info.trace_id: trace for trace in traces}

    trace1 = traces_by_id[trace_id_1]
    assert trace1.info.token_usage is not None
    assert trace1.info.token_usage["input_tokens"] == 100
    assert trace1.info.token_usage["output_tokens"] == 50
    assert trace1.info.token_usage["total_tokens"] == 150

    trace2 = traces_by_id[trace_id_2]
    assert trace2.info.token_usage is not None
    assert trace2.info.token_usage["input_tokens"] == 200
    assert trace2.info.token_usage["output_tokens"] == 100
    assert trace2.info.token_usage["total_tokens"] == 300

    trace3 = traces_by_id[trace_id_3]
    assert trace3.info.token_usage is None


def test_batch_get_trace_infos_basic(store: SqlAlchemyStore) -> None:
    from mlflow.tracing.constant import TraceMetadataKey

    experiment_id = store.create_experiment("test_batch_get_trace_infos")
    trace_id_1 = f"tr-{uuid.uuid4().hex}"
    trace_id_2 = f"tr-{uuid.uuid4().hex}"
    session_id = "session-123"

    # Create traces with session metadata
    trace_info_1 = TraceInfo(
        trace_id=trace_id_1,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceState.OK,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
    )
    store.start_trace(trace_info_1)

    trace_info_2 = TraceInfo(
        trace_id=trace_id_2,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        execution_duration=200,
        state=TraceState.OK,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
    )
    store.start_trace(trace_info_2)

    # Batch fetch trace infos
    trace_infos = store.batch_get_trace_infos([trace_id_1, trace_id_2])

    assert len(trace_infos) == 2
    trace_infos_by_id = {ti.trace_id: ti for ti in trace_infos}

    # Verify we got the trace infos
    assert trace_id_1 in trace_infos_by_id
    assert trace_id_2 in trace_infos_by_id

    # Verify metadata is present
    ti1 = trace_infos_by_id[trace_id_1]
    assert ti1.trace_id == trace_id_1
    assert ti1.timestamp_ms is not None
    assert ti1.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == session_id

    ti2 = trace_infos_by_id[trace_id_2]
    assert ti2.trace_id == trace_id_2
    assert ti2.timestamp_ms is not None
    assert ti2.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == session_id


def test_batch_get_trace_infos_empty(store: SqlAlchemyStore) -> None:
    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_infos = store.batch_get_trace_infos([trace_id])
    assert trace_infos == []


def test_batch_get_trace_infos_ordering(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_batch_get_trace_infos_ordering")
    trace_ids = [f"tr-{uuid.uuid4().hex}" for _ in range(3)]

    # Create traces in reverse order
    for i, trace_id in enumerate(reversed(trace_ids)):
        spans = [
            create_test_span(
                trace_id=trace_id,
                name=f"span_{i}",
                span_id=100 + i,
                status=trace_api.StatusCode.OK,
                start_ns=1_000_000_000 + i * 1_000_000_000,
                end_ns=2_000_000_000 + i * 1_000_000_000,
                trace_num=12345 + i,
            ),
        ]
        store.log_spans(experiment_id, spans)

    # Fetch in original order
    trace_infos = store.batch_get_trace_infos(trace_ids)

    # Verify order is preserved
    assert len(trace_infos) == 3
    for i, trace_info in enumerate(trace_infos):
        assert trace_info.trace_id == trace_ids[i]


def test_start_trace_creates_trace_metrics(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_start_trace_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        trace_metadata={
            TraceMetadataKey.TOKEN_USAGE: json.dumps({
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            })
        },
    )
    store.start_trace(trace_info)

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )

        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }


def test_start_trace_merge_preserves_existing_metrics(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_merge_preserves_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"
    loc = trace_location.TraceLocation.from_experiment_id(experiment_id)
    ts = get_current_time_millis()

    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=loc,
            request_time=ts,
            execution_duration=100,
            state=TraceStatus.OK,
            trace_metadata={
                TraceMetadataKey.TOKEN_USAGE: json.dumps({
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                })
            },
        )
    )

    # Second start_trace with a subset of metric keys triggers the merge path.
    result = store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=loc,
            request_time=ts,
            execution_duration=200,
            state=TraceStatus.OK,
            trace_metadata={
                TraceMetadataKey.TOKEN_USAGE: json.dumps({
                    "total_tokens": 110,
                    "cache_read_input_tokens": 5,
                })
            },
        )
    )

    assert result.trace_id == trace_id

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )
        metrics_by_key = {m.key: m.value for m in metrics}
        assert metrics_by_key == {
            "cache_read_input_tokens": 5,
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 110,
        }


def test_log_spans_creates_span_metrics(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        state=TraceStatus.OK,
    )
    store.start_trace(trace_info)

    otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }),
        SpanAttributeKey.MODEL: json.dumps("gpt-4-turbo"),
        SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
    }
    span = create_mlflow_span(otel_span, trace_id, "LLM")
    store.log_spans(experiment_id, [span])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlSpanMetrics)
            .filter(SqlSpanMetrics.trace_id == trace_id, SqlSpanMetrics.span_id == span.span_id)
            .order_by(SqlSpanMetrics.key)
            .all()
        )
        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }

        # Check that dimension_attributes is stored on the span
        sql_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_id, SqlSpan.span_id == span.span_id)
            .one()
        )
        assert sql_span.dimension_attributes[SpanAttributeKey.MODEL] == "gpt-4-turbo"
        assert sql_span.dimension_attributes[SpanAttributeKey.MODEL_PROVIDER] == "openai"


def test_log_spans_updates_trace_metrics_incrementally(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_incremental_metrics")
    trace_id = f"tr-{uuid.uuid4().hex}"

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")
    store.log_spans(experiment_id, [span1])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )

        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 200,
            "output_tokens": 75,
            "total_tokens": 275,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")
    store.log_spans(experiment_id, [span2])

    with store.ManagedSessionMaker() as session:
        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_id)
            .order_by(SqlTraceMetrics.key)
            .all()
        )
        metrics_by_key = {metric.key: metric.value for metric in metrics}
        assert metrics_by_key == {
            "input_tokens": 300,
            "output_tokens": 125,
            "total_tokens": 425,
        }


def test_log_spans_stores_span_metrics_per_span(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_log_spans_metrics_per_span")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        state=TraceStatus.OK,
    )
    store.start_trace(trace_info)

    otel_span1 = create_test_otel_span(
        trace_id=trace_id,
        name="first_llm_call",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id_num=12345,
        span_id_num=111,
    )
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            CostKey.INPUT_COST: 0.001,
            CostKey.OUTPUT_COST: 0.002,
            CostKey.TOTAL_COST: 0.003,
        }),
    }
    span1 = create_mlflow_span(otel_span1, trace_id, "LLM")

    otel_span2 = create_test_otel_span(
        trace_id=trace_id,
        name="second_llm_call",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        trace_id_num=12345,
        span_id_num=222,
    )
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        SpanAttributeKey.LLM_COST: json.dumps({
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }),
    }
    span2 = create_mlflow_span(otel_span2, trace_id, "LLM")

    store.log_spans(experiment_id, [span1, span2])

    with store.ManagedSessionMaker() as session:
        all_metrics = (
            session
            .query(SqlSpanMetrics)
            .filter(SqlSpanMetrics.trace_id == trace_id)
            .order_by(SqlSpanMetrics.span_id, SqlSpanMetrics.key)
            .all()
        )

        span1_metrics = {m.key: m.value for m in all_metrics if m.span_id == span1.span_id}
        assert span1_metrics == {
            CostKey.INPUT_COST: 0.001,
            CostKey.OUTPUT_COST: 0.002,
            CostKey.TOTAL_COST: 0.003,
        }

        span2_metrics = {m.key: m.value for m in all_metrics if m.span_id == span2.span_id}
        assert span2_metrics == {
            CostKey.INPUT_COST: 0.01,
            CostKey.OUTPUT_COST: 0.02,
            CostKey.TOTAL_COST: 0.03,
        }


def test_get_trace_basic(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_get_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    spans = [
        create_test_span(
            trace_id=trace_id,
            name="root_span",
            span_id=111,
            status=trace_api.StatusCode.OK,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="child_span",
            span_id=222,
            parent_id=111,
            status=trace_api.StatusCode.UNSET,
            start_ns=1_500_000_000,
            end_ns=1_800_000_000,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    trace = store.get_trace(trace_id)

    assert trace is not None
    loaded_spans = trace.data.spans

    assert len(loaded_spans) == 2

    root_span = next(s for s in loaded_spans if s.name == "root_span")
    child_span = next(s for s in loaded_spans if s.name == "child_span")

    assert root_span.trace_id == trace_id
    assert root_span.span_id == "000000000000006f"
    assert root_span.parent_id is None
    assert root_span.start_time_ns == 1_000_000_000
    assert root_span.end_time_ns == 2_000_000_000

    assert child_span.trace_id == trace_id
    assert child_span.span_id == "00000000000000de"
    assert child_span.parent_id == "000000000000006f"
    assert child_span.start_time_ns == 1_500_000_000
    assert child_span.end_time_ns == 1_800_000_000


def test_get_trace_not_found(store: SqlAlchemyStore) -> None:
    trace_id = f"tr-{uuid.uuid4().hex}"
    with pytest.raises(MlflowException, match=f"Trace with ID {trace_id} is not found."):
        store.get_trace(trace_id)


def test_start_trace_only_no_spans_location_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_start_trace_only")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    created_trace_info = store.start_trace(trace_info)

    assert TraceTagKey.SPANS_LOCATION not in created_trace_info.tags


def test_start_trace_then_log_spans_adds_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_start_trace_then_log_spans")
    trace_id = f"tr-{uuid.uuid4().hex}"

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info)

    span = create_test_span(
        trace_id=trace_id,
        name="test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )
    store.log_spans(experiment_id, [span])

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value


def test_log_spans_then_start_trace_preserves_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_log_spans_then_start_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    span = create_test_span(
        trace_id=trace_id,
        name="test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )
    store.log_spans(experiment_id, [span])

    trace_info_for_start = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info_for_start)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value


def test_log_spans_then_start_trace_preserves_preview(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_preview_preserved")
    trace_id = f"tr-{uuid.uuid4().hex}"

    span = create_test_span(
        trace_id=trace_id,
        name="llm_call",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
        attributes={
            "input.value": '{"messages": [{"role": "user", "content": "Hello"}]}',
            "output.value": '{"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}',
            "openinference.span.kind": "LLM",
        },
    )
    store.log_spans(experiment_id, [span])

    trace_info_for_start = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info_for_start)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.request_preview is not None
    assert trace_info.response_preview is not None
    assert "Hello" in trace_info.request_preview
    assert "Hi" in trace_info.response_preview


@pytest.mark.skipif(
    mlflow.get_tracking_uri().startswith("mysql"),
    reason="MySQL does not support concurrent log_spans calls for now",
)
def test_concurrent_log_spans_spans_location_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_concurrent_log_spans")
    trace_id = f"tr-{uuid.uuid4().hex}"

    def log_span_worker(span_id):
        span = create_test_span(
            trace_id=trace_id,
            name=f"concurrent_span_{span_id}",
            span_id=span_id,
            parent_id=111 if span_id != 111 else None,
            status=trace_api.StatusCode.OK,
            start_ns=1_000_000_000 + span_id * 1000,
            end_ns=2_000_000_000 + span_id * 1000,
            trace_num=12345,
        )
        store.log_spans(experiment_id, [span])
        return span_id

    # Simulate client-side workspace selection and ensure it propagates to worker threads.
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        # Launch multiple concurrent log_spans calls
        with ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="test-sqlalchemy-log-spans"
        ) as executor:
            futures = [executor.submit(log_span_worker, i) for i in range(111, 116)]

            # Wait for all to complete
            results = [future.result() for future in futures]

        # All workers should complete successfully
        assert len(results) == 5
        assert set(results) == {111, 112, 113, 114, 115}

        # Verify the SPANS_LOCATION tag was created correctly
        trace_info = store.get_trace_info(trace_id)
        assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value

        # Verify all spans were logged
        trace = store.get_trace(trace_id)
        assert len(trace.data.spans) == 5
        span_names = {span.name for span in trace.data.spans}
        expected_names = {f"concurrent_span_{i}" for i in range(111, 116)}
        assert span_names == expected_names


@pytest.mark.parametrize("allow_partial", [True, False])
def test_get_trace_with_partial_trace(store: SqlAlchemyStore, allow_partial: bool) -> None:
    experiment_id = store.create_experiment("test_partial_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Log only 1 span but indicate trace should have 2 spans
    spans = [
        create_test_span(
            trace_id=trace_id,
            name="span_1",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.SIZE_STATS: json.dumps({
                    TraceSizeStatsKey.NUM_SPANS: 2,  # Expecting 2 spans
                }),
            },
        )
    )

    if allow_partial:
        trace = store.get_trace(trace_id, allow_partial=allow_partial)
        assert trace is not None
        assert len(trace.data.spans) == 1
        assert trace.data.spans[0].name == "span_1"
    else:
        with pytest.raises(
            MlflowException,
            match=f"Trace with ID {trace_id} is not fully exported yet",
        ):
            store.get_trace(trace_id, allow_partial=allow_partial)


@pytest.mark.parametrize("allow_partial", [True, False])
def test_get_trace_with_complete_trace(store: SqlAlchemyStore, allow_partial: bool) -> None:
    experiment_id = store.create_experiment("test_complete_trace")
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Log 2 spans matching the expected count
    spans = [
        create_test_span(
            trace_id=trace_id,
            name="span_1",
            span_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
        create_test_span(
            trace_id=trace_id,
            name="span_2",
            span_id=222,
            parent_id=111,
            status=trace_api.StatusCode.OK,
            trace_num=12345,
        ),
    ]

    store.log_spans(experiment_id, spans)
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1234,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.SIZE_STATS: json.dumps({
                    TraceSizeStatsKey.NUM_SPANS: 2,  # Expecting 2 spans
                }),
            },
        )
    )

    # should always return the trace
    trace = store.get_trace(trace_id, allow_partial=allow_partial)
    assert trace is not None
    assert len(trace.data.spans) == 2


def test_log_spans_session_id_handling(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_session_id")

    # Session ID gets stored from span attributes
    trace_id1 = f"tr-{uuid.uuid4().hex}"
    otel_span1 = create_test_otel_span(trace_id=trace_id1)
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id1, cls=TraceJSONEncoder),
        "session.id": "session-123",
    }
    span1 = create_mlflow_span(otel_span1, trace_id1, "LLM")
    store.log_spans(experiment_id, [span1])

    trace_info1 = store.get_trace_info(trace_id1)
    assert trace_info1.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "session-123"

    # Existing session ID is preserved
    trace_id2 = f"tr-{uuid.uuid4().hex}"
    trace_with_session = TraceInfo(
        trace_id=trace_id2,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.IN_PROGRESS,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "existing-session"},
    )
    store.start_trace(trace_with_session)

    otel_span2 = create_test_otel_span(trace_id=trace_id2)
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id2, cls=TraceJSONEncoder),
        "session.id": "different-session",
    }
    span2 = create_mlflow_span(otel_span2, trace_id2, "LLM")
    store.log_spans(experiment_id, [span2])

    trace_info2 = store.get_trace_info(trace_id2)
    assert trace_info2.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "existing-session"

    # No session ID means no metadata
    trace_id3 = f"tr-{uuid.uuid4().hex}"
    otel_span3 = create_test_otel_span(trace_id=trace_id3)
    span3 = create_mlflow_span(otel_span3, trace_id3, "LLM")
    store.log_spans(experiment_id, [span3])

    trace_info3 = store.get_trace_info(trace_id3)
    assert TraceMetadataKey.TRACE_SESSION not in trace_info3.trace_metadata


def test_log_spans_user_id_handling(store: SqlAlchemyStore) -> None:
    experiment_id = store.create_experiment("test_user_id")

    # User ID gets stored from span attributes
    trace_id1 = f"tr-{uuid.uuid4().hex}"
    otel_span1 = create_test_otel_span(trace_id=trace_id1)
    otel_span1._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id1, cls=TraceJSONEncoder),
        "user.id": "alice",
    }
    span1 = create_mlflow_span(otel_span1, trace_id1, "LLM")
    store.log_spans(experiment_id, [span1])

    trace_info1 = store.get_trace_info(trace_id1)
    assert trace_info1.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "alice"

    # Existing user ID is preserved
    trace_id2 = f"tr-{uuid.uuid4().hex}"
    trace_with_user = TraceInfo(
        trace_id=trace_id2,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1234,
        execution_duration=100,
        state=TraceState.IN_PROGRESS,
        trace_metadata={TraceMetadataKey.TRACE_USER: "existing-user"},
    )
    store.start_trace(trace_with_user)

    otel_span2 = create_test_otel_span(trace_id=trace_id2)
    otel_span2._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id2, cls=TraceJSONEncoder),
        "user.id": "different-user",
    }
    span2 = create_mlflow_span(otel_span2, trace_id2, "LLM")
    store.log_spans(experiment_id, [span2])

    trace_info2 = store.get_trace_info(trace_id2)
    assert trace_info2.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "existing-user"

    # No user ID means no metadata
    trace_id3 = f"tr-{uuid.uuid4().hex}"
    otel_span3 = create_test_otel_span(trace_id=trace_id3)
    span3 = create_mlflow_span(otel_span3, trace_id3, "LLM")
    store.log_spans(experiment_id, [span3])

    trace_info3 = store.get_trace_info(trace_id3)
    assert TraceMetadataKey.TRACE_USER not in trace_info3.trace_metadata

    # Both session and user ID work together
    trace_id4 = f"tr-{uuid.uuid4().hex}"
    otel_span4 = create_test_otel_span(trace_id=trace_id4)
    otel_span4._attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id4, cls=TraceJSONEncoder),
        "session.id": "session-456",
        "user.id": "bob",
    }
    span4 = create_mlflow_span(otel_span4, trace_id4, "LLM")
    store.log_spans(experiment_id, [span4])

    trace_info4 = store.get_trace_info(trace_id4)
    assert trace_info4.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "session-456"
    assert trace_info4.trace_metadata.get(TraceMetadataKey.TRACE_USER) == "bob"


def test_find_completed_sessions(store: SqlAlchemyStore):
    """
    Test finding completed sessions based on their last trace timestamp.
    Sessions with last trace in time window are returned, ordered by last_trace_timestamp.
    """
    exp_id = store.create_experiment("test_find_completed_sessions")

    # Session A: last trace at t=2000
    for timestamp, trace_id in [(1000, "trace_a1"), (2000, "trace_a2")]:
        _create_trace(
            store,
            trace_id,
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
        )

    # Session B: last trace at t=4000
    for timestamp, trace_id in [(3000, "trace_b1"), (4000, "trace_b2")]:
        _create_trace(
            store,
            trace_id,
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-b"},
        )

    # Session C: last trace at t=10000 (outside query window)
    for timestamp, trace_id in [(5000, "trace_c1"), (10000, "trace_c2")]:
        _create_trace(
            store,
            trace_id,
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-c"},
        )

    _create_trace(store, "trace_no_session", exp_id, request_time=2500)

    # Query window [0, 5000] should return session-a and session-b
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=5000,
    )

    assert len(completed) == 2
    assert {s.session_id for s in completed} == {"session-a", "session-b"}
    assert completed[0].session_id == "session-a"
    assert completed[0].first_trace_timestamp_ms == 1000
    assert completed[0].last_trace_timestamp_ms == 2000
    assert completed[1].session_id == "session-b"
    assert completed[1].first_trace_timestamp_ms == 3000
    assert completed[1].last_trace_timestamp_ms == 4000

    # Narrower window [3000, 5000] should only return session-b
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=3000,
        max_last_trace_timestamp_ms=5000,
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-b"

    # Test max_results pagination
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=5000,
        max_results=1,
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-a"


def test_find_completed_sessions_aggregates_across_all_traces(store: SqlAlchemyStore):
    """
    Regression test: first/last timestamps should be computed across ALL session traces,
    not just those matching the min_last_trace_timestamp_ms filter.
    """
    exp_id = store.create_experiment("test_session_timestamp_aggregation")

    _create_trace(
        store,
        "trace1",
        exp_id,
        request_time=1000,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
    )
    _create_trace(
        store,
        "trace2",
        exp_id,
        request_time=3000,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-a"},
    )

    completed = store.find_completed_sessions(
        experiment_id=exp_id, min_last_trace_timestamp_ms=2000, max_last_trace_timestamp_ms=4000
    )

    assert len(completed) == 1
    assert completed[0].first_trace_timestamp_ms == 1000
    assert completed[0].last_trace_timestamp_ms == 3000


def test_find_completed_sessions_with_filter_string(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_find_completed_sessions_with_filter")

    # Session A: first trace env="prod", second env="dev" - should match prod filter
    # Session B: first trace env="dev", second env="prod" - should NOT match prod filter
    for session_id, times, envs in [
        ("session-a", [1000, 2000], ["prod", "dev"]),
        ("session-b", [3000, 4000], ["dev", "prod"]),
    ]:
        for timestamp, env in zip(times, envs):
            _create_trace(
                store,
                f"trace_{session_id}_{timestamp}",
                exp_id,
                request_time=timestamp,
                trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
                tags={"env": env},
            )

    # Tag filter should only match session-a (first trace has env=prod)
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=10000,
        filter_string="tag.env = 'prod'",
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-a"
    assert completed[0].first_trace_timestamp_ms == 1000
    assert completed[0].last_trace_timestamp_ms == 2000

    # Session C: test metadata filter (first trace user_id="alice", second user_id="bob")
    for timestamp, user in [(5000, "alice"), (6000, "bob")]:
        _create_trace(
            store,
            f"trace_c_{timestamp}",
            exp_id,
            request_time=timestamp,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-c", "user_id": user},
        )

    # Metadata filter should match session-c (first trace has user_id=alice)
    completed = store.find_completed_sessions(
        experiment_id=exp_id,
        min_last_trace_timestamp_ms=0,
        max_last_trace_timestamp_ms=10000,
        filter_string="metadata.user_id = 'alice'",
    )
    assert len(completed) == 1
    assert completed[0].session_id == "session-c"


def _archive_traces(
    store: SqlAlchemyStore,
    *,
    default_trace_archival_location: str,
    default_retention: str,
    long_retention_allowlist: set[str] | list[str] | None = None,
    max_traces: int | None = 100,
    now_millis: int | None = None,
) -> int:
    kwargs = {
        "default_trace_archival_location": default_trace_archival_location,
        "default_retention": default_retention,
        "long_retention_allowlist": long_retention_allowlist,
        "max_traces": max_traces,
    }
    if now_millis is None:
        return store.archive_traces(**kwargs)

    with mock.patch.object(store, "_get_archive_traces_now_millis", return_value=now_millis):
        return store.archive_traces(**kwargs)


@pytest.fixture
def test_log_spans_rejects_archived_trace_in_multi_trace_batch(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_multi_trace_archived_experiment")
    archived_trace_id = "tr-multi-archived"
    fresh_trace_id = "tr-multi-fresh"

    store.log_spans(
        experiment_id,
        [
            create_test_span(
                archived_trace_id,
                span_id=1,
                start_ns=1_000_000_000,
                end_ns=2_000_000_000,
            )
        ],
    )
    store.set_trace_tag(
        archived_trace_id, TraceTagKey.SPANS_LOCATION, SpansLocation.ARCHIVE_REPO.value
    )
    store.set_trace_tag(archived_trace_id, TraceTagKey.ARCHIVE_LOCATION, "dbfs:/archive/tr-multi")

    with pytest.raises(
        MlflowException, match=f"Cannot log spans to archived traces: '{archived_trace_id}'"
    ) as exc_info:
        store.log_spans(
            experiment_id,
            [
                create_test_span(
                    archived_trace_id,
                    span_id=2,
                    start_ns=3_000_000_000,
                    end_ns=4_000_000_000,
                ),
                create_test_span(
                    fresh_trace_id,
                    span_id=3,
                    start_ns=5_000_000_000,
                    end_ns=6_000_000_000,
                ),
            ],
        )

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlSpan).filter_by(trace_id=fresh_trace_id).count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_creates_trace_if_not_exists(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment but no trace
    experiment_id = store.create_experiment("test_auto_trace_experiment")

    # Create a span without a pre-existing trace
    trace_id = "tr-auto-created-trace"
    readable_span = OTelReadableSpan(
        name="auto_trace_span",
        context=trace_api.SpanContext(
            trace_id=98765,
            span_id=555,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
            "mlflow.experimentId": json.dumps(experiment_id, cls=TraceJSONEncoder),
        },
        start_time=5000000000,
        end_time=6000000000,
        resource=_OTelResource.get_empty(),
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Log the span - should create the trace automatically
    if is_async:
        logged_spans = await store.log_spans_async(experiment_id, [span])
    else:
        logged_spans = store.log_spans(experiment_id, [span])

    assert len(logged_spans) == 1
    assert logged_spans[0] == span

    # Verify the trace was created
    with store.ManagedSessionMaker() as session:
        created_trace = (
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).first()
        )

        assert created_trace is not None
        assert created_trace.experiment_id == int(experiment_id)
        assert created_trace.timestamp_ms == 5000000000 // 1_000_000
        assert created_trace.execution_time_ms == 1000000000 // 1_000_000
        # When root span status is UNSET (unexpected), we assume trace status is OK
        assert created_trace.status == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_empty_list(store: SqlAlchemyStore, is_async: bool):
    experiment_id = store.create_experiment("test_empty_experiment")

    if is_async:
        result = await store.log_spans_async(experiment_id, [])
    else:
        result = store.log_spans(experiment_id, [])
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_concurrent_trace_creation(store: SqlAlchemyStore, is_async: bool):
    # Create an experiment
    experiment_id = store.create_experiment("test_concurrent_trace")
    trace_id = "tr-concurrent-test"

    # Create a span
    readable_span = OTelReadableSpan(
        name="concurrent_span",
        context=trace_api.SpanContext(
            trace_id=12345,
            span_id=999,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        ),
        parent=None,
        resource=_OTelResource.get_empty(),
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
        },
        start_time=1000000000,
        end_time=2000000000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        events=[],
        links=[],
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Simulate a race condition where flush() raises IntegrityError
    # This tests that the code properly handles concurrent trace creation
    original_flush = None
    call_count = 0

    def mock_flush(self):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call to flush (for trace creation) raises IntegrityError
            raise IntegrityError("UNIQUE constraint failed", None, None)
        else:
            # Subsequent calls work normally
            return original_flush()

    with store.ManagedSessionMaker() as session:
        original_flush = session.flush
        with mock.patch.object(session, "flush", mock_flush):
            # This should handle the IntegrityError and still succeed
            if is_async:
                result = await store.log_spans_async(experiment_id, [span])
            else:
                result = store.log_spans(experiment_id, [span])

    # Verify the span was logged successfully despite the race condition
    assert len(result) == 1
    assert result[0] == span

    # Verify the trace and span exist in the database
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.experiment_id == int(experiment_id)

        saved_span = (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_id, SqlSpan.span_id == span.span_id)
            .one()
        )
        assert saved_span is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_trace_time_range(store: SqlAlchemyStore, is_async: bool):
    experiment_id = _create_experiments(store, "test_log_spans_updates_trace")
    trace_id = "tr-time-update-test-123"

    # Create first span from 1s to 2s
    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="early_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=1_000_000_000,  # 1 second in nanoseconds
            end_time=2_000_000_000,  # 2 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log first span - creates trace with 1s start, 1s duration
    if is_async:
        await store.log_spans_async(experiment_id, [span1])
    else:
        store.log_spans(experiment_id, [span1])

    # Verify initial trace times
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 1_000  # 1 second
        assert trace.execution_time_ms == 1_000  # 1 second duration

    # Create second span that starts earlier (0.5s) and ends later (3s)
    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="extended_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=500_000_000,  # 0.5 seconds
            end_time=3_000_000_000,  # 3 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log second span - should update trace to 0.5s start, 2.5s duration
    if is_async:
        await store.log_spans_async(experiment_id, [span2])
    else:
        store.log_spans(experiment_id, [span2])

    # Verify trace times were updated
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # 0.5 seconds (earlier start)
        assert trace.execution_time_ms == 2_500  # 2.5 seconds duration (0.5s to 3s)

    # Create third span that only extends the end time (2.5s to 4s)
    span3 = create_mlflow_span(
        OTelReadableSpan(
            name="later_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=333,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=2_500_000_000,  # 2.5 seconds
            end_time=4_000_000_000,  # 4 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log third span - should only update end time
    if is_async:
        await store.log_spans_async(experiment_id, [span3])
    else:
        store.log_spans(experiment_id, [span3])

    # Verify trace times were updated again
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Still 0.5 seconds (no earlier start)
        assert trace.execution_time_ms == 3_500  # 3.5 seconds duration (0.5s to 4s)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_no_end_time(store: SqlAlchemyStore, is_async: bool):
    experiment_id = _create_experiments(store, "test_log_spans_no_end_time")
    trace_id = "tr-no-end-time-test-123"

    # Create span without end time (in-progress span)
    span1 = create_mlflow_span(
        OTelReadableSpan(
            name="in_progress_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=1_000_000_000,  # 1 second in nanoseconds
            end_time=None,  # No end time - span still in progress
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log span with no end time
    if is_async:
        await store.log_spans_async(experiment_id, [span1])
    else:
        store.log_spans(experiment_id, [span1])

    # Verify trace has timestamp but no execution_time
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 1_000  # 1 second
        assert trace.execution_time_ms is None  # No execution time since span not ended

    # Add a second span that also has no end time
    span2 = create_mlflow_span(
        OTelReadableSpan(
            name="another_in_progress_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=500_000_000,  # 0.5 seconds - earlier start
            end_time=None,  # No end time
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log second span with no end time
    if is_async:
        await store.log_spans_async(experiment_id, [span2])
    else:
        store.log_spans(experiment_id, [span2])

    # Verify trace timestamp updated but execution_time still None
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Updated to earlier time
        assert trace.execution_time_ms is None  # Still no execution time

    # Now add a span with an end time
    span3 = create_mlflow_span(
        OTelReadableSpan(
            name="completed_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=333,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={"mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder)},
            start_time=2_000_000_000,  # 2 seconds
            end_time=3_000_000_000,  # 3 seconds
            resource=_OTelResource.get_empty(),
        ),
        trace_id,
    )

    # Log span with end time
    if is_async:
        await store.log_spans_async(experiment_id, [span3])
    else:
        store.log_spans(experiment_id, [span3])

    # Verify trace now has execution_time
    with store.ManagedSessionMaker() as session:
        trace = session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).one()
        assert trace.timestamp_ms == 500  # Still earliest start
        assert trace.execution_time_ms == 2_500  # 3s - 0.5s = 2.5s


def test_log_spans_then_start_trace_preserves_archived_trace_tags(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_preserve_archived_trace_tags")
    trace_id = f"tr-{uuid.uuid4().hex}"
    archive_location = "s3://bucket/archive/traces.pb"

    span = create_test_span(
        trace_id=trace_id,
        name="test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )
    store.log_spans(experiment_id, [span])
    store.set_trace_tag(trace_id, TraceTagKey.SPANS_LOCATION, SpansLocation.ARCHIVE_REPO.value)
    store.set_trace_tag(trace_id, TraceTagKey.ARCHIVE_LOCATION, archive_location)

    trace_info_for_start = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info_for_start)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
    assert trace_info.tags[TraceTagKey.ARCHIVE_LOCATION] == archive_location


def test_log_spans_then_start_trace_preserves_archival_failure_tag(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_preserve_archival_failure_tag")
    trace_id = f"tr-{uuid.uuid4().hex}"

    span = create_test_span(
        trace_id=trace_id,
        name="test_span",
        span_id=111,
        status=trace_api.StatusCode.OK,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        trace_num=12345,
    )
    store.log_spans(experiment_id, [span])
    store.set_trace_tag(
        trace_id,
        TraceTagKey.ARCHIVAL_FAILURE,
        TraceArchivalFailureReason.MALFORMED_TRACE.value,
    )

    trace_info_for_start = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
        tags={"custom_tag": "value"},
        trace_metadata={"source": "test"},
    )
    store.start_trace(trace_info_for_start)

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value
    assert trace_info.tags[TraceTagKey.ARCHIVAL_FAILURE] == (
        TraceArchivalFailureReason.MALFORMED_TRACE.value
    )


def test_archive_traces_archives_db_backed_trace_payloads(
    store: SqlAlchemyStore, workspaces_enabled: bool
):
    exp_id = store.create_experiment("archive-db-backed")
    now_millis = 10 * 24 * 60 * 60 * 1000
    old_trace_id = "tr-archive-old"
    new_trace_id = "tr-archive-new"
    old_request_time = now_millis - 3 * 24 * 60 * 60 * 1000  # 3 days old
    new_request_time = now_millis - 60 * 60 * 1000  # 1 hour old

    _create_trace(store, old_trace_id, exp_id, request_time=old_request_time)
    _create_trace(store, new_trace_id, exp_id, request_time=new_request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                old_trace_id,
                span_id=111,
                start_ns=old_request_time * 1_000_000,
                end_ns=(old_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_id,
        [
            create_test_span(
                new_trace_id,
                span_id=222,
                start_ns=new_request_time * 1_000_000,
                end_ns=(new_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    original_trace_artifact_uri = store.get_trace_info(old_trace_id).tags[MLFLOW_ARTIFACT_LOCATION]

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )
        assert archived == 1

        trace_info = store.get_trace_info(old_trace_id)
        assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
        if workspaces_enabled:
            expected_archive_uri = append_to_uri_path(
                archive_root.as_uri(),
                WORKSPACES_DIR_NAME,
                DEFAULT_WORKSPACE_NAME,
                exp_id,
                SqlAlchemyStore.TRACE_FOLDER_NAME,
                old_trace_id,
                SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
            )
            expected_archive_path = (
                archive_root
                / WORKSPACES_DIR_NAME
                / DEFAULT_WORKSPACE_NAME
                / exp_id
                / SqlAlchemyStore.TRACE_FOLDER_NAME
                / old_trace_id
                / SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
                / "traces.pb"
            )
        else:
            expected_archive_uri = append_to_uri_path(
                archive_root.as_uri(),
                exp_id,
                SqlAlchemyStore.TRACE_FOLDER_NAME,
                old_trace_id,
                SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
            )
            expected_archive_path = (
                archive_root
                / exp_id
                / SqlAlchemyStore.TRACE_FOLDER_NAME
                / old_trace_id
                / SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
                / "traces.pb"
            )
        assert trace_info.tags[MLFLOW_ARTIFACT_LOCATION] == original_trace_artifact_uri
        assert trace_info.tags[TraceTagKey.ARCHIVE_LOCATION] == expected_archive_uri
        assert expected_archive_path.is_file()

        with store.ManagedSessionMaker() as session:
            archived_span = session.query(SqlSpan).filter(SqlSpan.trace_id == old_trace_id).one()
            fresh_span = session.query(SqlSpan).filter(SqlSpan.trace_id == new_trace_id).one()
            assert archived_span.content == ""
            assert fresh_span.content != ""

        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        archived_trace_data = get_artifact_repository(
            trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]
        ).download_archived_trace_data()
        assert len(archived_trace_data.spans) == 1
        assert archived_trace_data.spans[0].name == "test_span"
        assert store.get_trace(old_trace_id).data.spans[0].name == "test_span"
        assert store.batch_get_traces([old_trace_id])[0].data.spans[0].name == "test_span"

        assert (
            _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="1d",
                now_millis=now_millis,
            )
            == 0
        )


def test_batch_get_traces_raises_for_archived_traces_with_missing_payload(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-batch-get-missing-payload")
    archived_trace_id = "tr-archive-missing-payload"
    healthy_trace_id = "tr-archive-batch-healthy"
    now_millis = 25 * 24 * 60 * 60 * 1000
    archived_request_time = now_millis - 2 * 24 * 60 * 60 * 1000
    healthy_request_time = now_millis - 12 * 60 * 60 * 1000

    _create_trace(store, archived_trace_id, exp_id, request_time=archived_request_time)
    _create_trace(store, healthy_trace_id, exp_id, request_time=healthy_request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                archived_trace_id,
                span_id=151,
                start_ns=archived_request_time * 1_000_000,
                end_ns=(archived_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_id,
        [
            create_test_span(
                healthy_trace_id,
                span_id=152,
                start_ns=healthy_request_time * 1_000_000,
                end_ns=(healthy_request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME

        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )

        assert archived == 1
        archived_trace_info = store.get_trace_info(archived_trace_id)
        archive_payload_path = (
            Path(local_file_uri_to_path(archived_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]))
            / TRACE_ARCHIVAL_FILENAME
        )
        assert archive_payload_path.is_file()
        archive_payload_path.unlink()

        with pytest.raises(MlflowTraceDataNotFound, match="Trace data not found"):
            store.batch_get_traces([archived_trace_id, healthy_trace_id])


def test_archived_trace_with_spanless_payload_raises_corruption(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-spanless-payload")
    trace_id = "tr-archive-spanless-payload"
    now_millis = 25 * 24 * 60 * 60 * 1000
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=151,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

        from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME

        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )

        assert archived == 1
        archived_trace_info = store.get_trace_info(trace_id)
        archive_payload_path = (
            Path(local_file_uri_to_path(archived_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]))
            / TRACE_ARCHIVAL_FILENAME
        )
        assert archive_payload_path.is_file()

        traces_data = TracesData()
        traces_data.resource_spans.add().scope_spans.add()
        archive_payload_path.write_bytes(traces_data.SerializeToString())

        with pytest.raises(MlflowTraceDataCorrupted, match="Trace data is corrupted"):
            store.get_trace(trace_id)
        with pytest.raises(MlflowTraceDataCorrupted, match="Trace data is corrupted"):
            store.batch_get_traces([trace_id])


def test_archive_traces_preserve_root_first_span_order(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive-span-order")
    trace_id = "tr-archive-span-order"
    now_millis = 12 * 24 * 60 * 60 * 1000
    request_time = now_millis - 3 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                name="root",
                span_id=999,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            ),
            create_test_span(
                trace_id,
                name="child",
                span_id=111,
                parent_id=999,
                start_ns=(request_time + 1) * 1_000_000,
                end_ns=(request_time + 900) * 1_000_000,
            ),
        ],
    )

    assert [span.name for span in store.get_trace(trace_id).data.spans] == ["root", "child"]

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )
        assert archived == 1
        assert [span.name for span in store.get_trace(trace_id).data.spans] == ["root", "child"]
        assert [span.name for span in store.batch_get_traces([trace_id])[0].data.spans] == [
            "root",
            "child",
        ]


def test_archive_traces_preserves_trace_attachment_location(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive-attachments")
    trace_id = "tr-archive-attachments"
    now_millis = 25 * 24 * 60 * 60 * 1000
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000
    attachment_id = str(uuid.uuid4())
    attachment_bytes = b"attachment-bytes"

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=919,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )
    trace_info = store.get_trace_info(trace_id)

    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

    get_artifact_repository(trace_info.tags[MLFLOW_ARTIFACT_LOCATION]).upload_attachment(
        attachment_id, attachment_bytes
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )

    assert archived == 1
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[MLFLOW_ARTIFACT_LOCATION].endswith(
        f"/{exp_id}/traces/{trace_id}/artifacts"
    )
    assert TraceTagKey.ARCHIVE_LOCATION in trace_info.tags
    assert (
        get_artifact_repository(
            trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
        ).download_trace_attachment(attachment_id)
        == attachment_bytes
    )


def test_archive_traces_raises_when_default_root_is_unset_and_no_workspace_override(
    store: SqlAlchemyStore, workspaces_enabled: bool
):
    if workspaces_enabled:
        workspace_store = store._get_workspace_provider_instance()
        workspace_store.update_workspace(
            Workspace(
                name=DEFAULT_WORKSPACE_NAME,
                trace_archival_location="",
            )
        )

    exp_id = store.create_experiment("archive-default-root")
    now_millis = 15 * 24 * 60 * 60 * 1000
    trace_id = "tr-archive-default-root"
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000  # 2 days old

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=113,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )
    with pytest.raises(MlflowException, match="default_trace_archival_location") as exc_info:
        _archive_traces(
            store,
            default_trace_archival_location=None,
            default_retention="1d",
            now_millis=now_millis,
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_archive_traces_raises_when_default_retention_is_unset(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="default_retention") as exc_info:
        store.archive_traces(
            default_trace_archival_location="s3://archive/default",
            default_retention=None,
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_archive_traces_rejects_proxy_only_default_root(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="proxy-only `mlflow-artifacts:` scheme") as exc_info:
        store.archive_traces(
            default_trace_archival_location="mlflow-artifacts:/archive/default",
            default_retention="1d",
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_archive_traces_rejects_unregistered_archive_scheme_before_processing_candidates(
    store: SqlAlchemyStore,
):
    with pytest.raises(MlflowException, match="Could not find a registered artifact repository"):
        store.archive_traces(
            default_trace_archival_location="unknown-scheme://archive/default",
            default_retention="1d",
        )


def test_archive_traces_raises_when_default_retention_exceeds_max_length(
    store: SqlAlchemyStore,
):
    with pytest.raises(MlflowException, match="at most 32 characters") as exc_info:
        store.archive_traces(
            default_trace_archival_location="s3://archive/default",
            default_retention=f"{'1' * 32}d",
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_archive_traces_treats_unset_max_traces_as_unbounded(store: SqlAlchemyStore, monkeypatch):
    store.create_experiment("archive-unbounded")
    archived_trace_ids = []
    candidates = [
        _TraceArchiveCandidate(
            trace_id=f"tr-unbounded-{idx}",
            experiment_id="0",
            timestamp_ms=idx,
        )
        for idx in range(101)
    ]

    def _capture_find_archivable_trace_candidates_for_experiments(
        *, session, experiment_ids, max_timestamp_millis, limit
    ):
        assert limit is None
        return candidates

    def _capture_archive_trace_candidate(*, trace_id, trace_archival_config):
        archived_trace_ids.append(trace_id)
        return True

    monkeypatch.setattr(
        store,
        "_find_archivable_trace_candidates_for_experiments",
        _capture_find_archivable_trace_candidates_for_experiments,
    )
    monkeypatch.setattr(
        store,
        "_archive_trace_candidate",
        _capture_archive_trace_candidate,
    )

    archived = store.archive_traces(
        default_trace_archival_location="s3://archive/default",
        default_retention="30d",
    )

    assert archived == len(candidates)
    assert archived_trace_ids == [candidate.trace_id for candidate in candidates]


def test_archive_traces_raises_internal_error_when_resolution_returns_no_location(
    store: SqlAlchemyStore, monkeypatch
):
    def _broken_resolve_trace_archival_config(
        *, default_trace_archival_location, default_retention
    ):
        return ResolvedTraceArchivalConfig(
            config=TraceArchivalConfig(location=None, retention=default_retention),
            append_workspace_prefix=False,
        )

    monkeypatch.setattr(
        store,
        "resolve_trace_archival_config",
        _broken_resolve_trace_archival_config,
    )

    with pytest.raises(
        MlflowException, match="config resolution returned no archival location"
    ) as exc_info:
        store.archive_traces(
            default_trace_archival_location="s3://archive/default",
            default_retention="1d",
        )
    assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_archive_traces_raises_internal_error_when_resolution_returns_no_retention(
    store: SqlAlchemyStore, monkeypatch
):
    def _broken_resolve_trace_archival_config(
        *, default_trace_archival_location, default_retention
    ):
        return ResolvedTraceArchivalConfig(
            config=TraceArchivalConfig(
                location=default_trace_archival_location,
                retention=None,
            ),
            append_workspace_prefix=False,
        )

    monkeypatch.setattr(
        store,
        "resolve_trace_archival_config",
        _broken_resolve_trace_archival_config,
    )

    with pytest.raises(
        MlflowException, match="config resolution returned no archival retention"
    ) as exc_info:
        store.archive_traces(
            default_trace_archival_location="s3://archive/default",
            default_retention="1d",
        )
    assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_archive_traces_respects_experiment_retention_and_archive_now(store: SqlAlchemyStore):
    now_millis = 20 * 24 * 60 * 60 * 1000
    exp_short_retention = store.create_experiment("archive-short-retention")
    exp_archive_now = store.create_experiment("archive-now")
    short_old_request_time = now_millis - 2 * 24 * 60 * 60 * 1000  # 2 days old
    short_new_request_time = now_millis - 12 * 60 * 60 * 1000  # 12 hours old
    now_old_request_time = now_millis - 2 * 24 * 60 * 60 * 1000  # 2 days old
    now_new_request_time = now_millis - 2 * 60 * 60 * 1000  # 2 hours old

    store.set_experiment_tag(
        exp_short_retention,
        ExperimentTag(
            TraceExperimentTagKey.ARCHIVAL_RETENTION,
            json.dumps({"type": "duration", "value": "1d"}),
        ),
    )
    store.set_experiment_tag(
        exp_archive_now,
        ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({"older_than": "1d"})),
    )

    _create_trace(
        store,
        "tr-short-old",
        exp_short_retention,
        request_time=short_old_request_time,
    )
    _create_trace(
        store,
        "tr-short-new",
        exp_short_retention,
        request_time=short_new_request_time,
    )
    _create_trace(
        store,
        "tr-now-old",
        exp_archive_now,
        request_time=now_old_request_time,
    )
    _create_trace(
        store,
        "tr-now-new",
        exp_archive_now,
        request_time=now_new_request_time,
    )
    store.log_spans(
        exp_short_retention,
        [
            create_test_span(
                "tr-short-old",
                span_id=111,
                start_ns=short_old_request_time * 1_000_000,
                end_ns=(short_old_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_short_retention,
        [
            create_test_span(
                "tr-short-new",
                span_id=112,
                start_ns=short_new_request_time * 1_000_000,
                end_ns=(short_new_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_archive_now,
        [
            create_test_span(
                "tr-now-old",
                span_id=211,
                start_ns=now_old_request_time * 1_000_000,
                end_ns=(now_old_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_archive_now,
        [
            create_test_span(
                "tr-now-new",
                span_id=212,
                start_ns=now_new_request_time * 1_000_000,
                end_ns=(now_new_request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="30d",
            now_millis=now_millis,
        )

    assert archived == 2
    assert store.get_trace_info("tr-short-old").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-now-old").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-short-new").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )
    assert store.get_trace_info("tr-now-new").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_archive_now).tags


def test_archive_traces_skips_regular_pass_when_archive_now_covers_retention(
    store: SqlAlchemyStore, monkeypatch
):
    now_millis = 20 * 24 * 60 * 60 * 1000
    day_millis = 24 * 60 * 60 * 1000
    exp_id = store.create_experiment("archive-now-covers-retention")
    find_calls = []

    store.set_experiment_tag(
        exp_id,
        ExperimentTag(
            TraceExperimentTagKey.ARCHIVAL_RETENTION,
            json.dumps({"type": "duration", "value": "7d"}),
        ),
    )
    store.set_experiment_tag(
        exp_id,
        ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({"older_than": "1d"})),
    )

    def _capture_find_archivable_trace_candidates_for_experiments(
        *, session, experiment_ids, max_timestamp_millis, limit
    ):
        find_calls.append((tuple(experiment_ids), max_timestamp_millis, limit))
        return []

    monkeypatch.setattr(
        store,
        "_find_archivable_trace_candidates_for_experiments",
        _capture_find_archivable_trace_candidates_for_experiments,
    )

    archived = _archive_traces(
        store,
        default_trace_archival_location="s3://archive/default",
        default_retention="30d",
        now_millis=now_millis,
    )

    assert archived == 0
    assert ((exp_id,), now_millis - day_millis, 100) in find_calls
    assert all(
        exp_id not in experiment_ids or max_timestamp_millis != now_millis - 7 * day_millis
        for experiment_ids, max_timestamp_millis, _ in find_calls
    )


def test_archive_traces_keeps_oldest_archive_now_candidates_when_bounded(
    store: SqlAlchemyStore, monkeypatch
):
    exp_older = store.create_experiment("archive-now-bounded-older")
    exp_newer = store.create_experiment("archive-now-bounded-newer")
    archived_trace_ids = []

    for exp_id in (exp_older, exp_newer):
        store.set_experiment_tag(
            exp_id,
            ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({})),
        )

    grouped_candidates = [
        _TraceArchiveCandidate(
            trace_id="tr-oldest",
            experiment_id=exp_newer,
            timestamp_ms=10,
        ),
        _TraceArchiveCandidate(
            trace_id="tr-second-oldest",
            experiment_id=exp_older,
            timestamp_ms=20,
        ),
        _TraceArchiveCandidate(
            trace_id="tr-third-oldest",
            experiment_id=exp_newer,
            timestamp_ms=30,
        ),
        _TraceArchiveCandidate(
            trace_id="tr-fourth-oldest",
            experiment_id=exp_older,
            timestamp_ms=40,
        ),
    ]

    def _capture_find_archivable_trace_candidates_for_experiments(
        *, session, experiment_ids, max_timestamp_millis, limit
    ):
        assert experiment_ids == [exp_older, exp_newer]
        assert max_timestamp_millis is None
        assert limit == 2
        return grouped_candidates

    def _capture_archive_trace_candidate(*, trace_id, trace_archival_config):
        archived_trace_ids.append(trace_id)
        return True

    monkeypatch.setattr(
        store,
        "_find_archivable_trace_candidates_for_experiments",
        _capture_find_archivable_trace_candidates_for_experiments,
    )
    monkeypatch.setattr(
        store,
        "_archive_trace_candidate",
        _capture_archive_trace_candidate,
    )

    archived = store.archive_traces(
        default_trace_archival_location="s3://archive/default",
        default_retention="30d",
        max_traces=2,
    )

    assert archived == 2
    assert archived_trace_ids == ["tr-oldest", "tr-second-oldest"]


def test_archive_traces_groups_regular_candidate_queries_by_shared_cutoff(
    store: SqlAlchemyStore, monkeypatch
):
    now_millis = 20 * 24 * 60 * 60 * 1000
    exp_first = store.create_experiment("archive-regular-grouped-first")
    exp_second = store.create_experiment("archive-regular-grouped-second")
    find_calls = []

    def _capture_find_archivable_trace_candidates_for_experiments(
        *, session, experiment_ids, max_timestamp_millis, limit
    ):
        find_calls.append((tuple(experiment_ids), max_timestamp_millis, limit))
        return []

    monkeypatch.setattr(
        store,
        "_find_archivable_trace_candidates_for_experiments",
        _capture_find_archivable_trace_candidates_for_experiments,
    )

    archived = _archive_traces(
        store,
        default_trace_archival_location="s3://archive/default",
        default_retention="30d",
        now_millis=now_millis,
    )

    assert archived == 0
    assert len(find_calls) == 1
    experiment_ids, max_timestamp_millis, limit = find_calls[0]
    assert set(experiment_ids) == {"0", exp_first, exp_second}
    assert max_timestamp_millis == now_millis - 30 * 24 * 60 * 60 * 1000
    assert limit == 100


def test_archive_traces_chunks_large_experiment_groups_and_keeps_oldest_candidates(
    store: SqlAlchemyStore, monkeypatch
):
    monkeypatch.setattr(sqlalchemy_store_module, "_TRACE_ARCHIVAL_EXPERIMENT_ID_CHUNK_SIZE", 2)

    now_millis = 40 * 24 * 60 * 60 * 1000
    exp_recent = store.create_experiment("archive-chunked-recent")
    exp_oldest = store.create_experiment("archive-chunked-oldest")
    exp_second_oldest = store.create_experiment("archive-chunked-second-oldest")

    trace_configs = [
        (exp_recent, "tr-chunked-recent", now_millis - 3 * 24 * 60 * 60 * 1000, 811),
        (exp_oldest, "tr-chunked-oldest", now_millis - 5 * 24 * 60 * 60 * 1000, 812),
        (
            exp_second_oldest,
            "tr-chunked-second-oldest",
            now_millis - 4 * 24 * 60 * 60 * 1000,
            813,
        ),
    ]
    for exp_id, trace_id, request_time, span_id in trace_configs:
        _create_trace(store, trace_id, exp_id, request_time=request_time)
        store.log_spans(
            exp_id,
            [
                create_test_span(
                    trace_id,
                    span_id=span_id,
                    start_ns=request_time * 1_000_000,
                    end_ns=(request_time + 1_000) * 1_000_000,
                )
            ],
        )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            max_traces=2,
            now_millis=now_millis,
        )

    assert archived == 2
    assert store.get_trace_info("tr-chunked-oldest").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-chunked-second-oldest").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-chunked-recent").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )


def test_archive_traces_keeps_regular_pass_when_archive_now_is_narrower(
    store: SqlAlchemyStore,
):
    day_millis = 24 * 60 * 60 * 1000
    now_millis = 60 * day_millis
    exp_id = store.create_experiment("archive-now-narrower-than-retention")
    archive_now_trace_time = now_millis - 40 * day_millis
    retention_trace_time = now_millis - 10 * day_millis
    recent_trace_time = now_millis - 2 * day_millis

    store.set_experiment_tag(
        exp_id,
        ExperimentTag(
            TraceExperimentTagKey.ARCHIVAL_RETENTION,
            json.dumps({"type": "duration", "value": "7d"}),
        ),
    )
    store.set_experiment_tag(
        exp_id,
        ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({"older_than": "30d"})),
    )

    for trace_id, request_time in (
        ("tr-now-priority", archive_now_trace_time),
        ("tr-now-retention", retention_trace_time),
        ("tr-now-recent", recent_trace_time),
    ):
        _create_trace(store, trace_id, exp_id, request_time=request_time)
        store.log_spans(
            exp_id,
            [
                create_test_span(
                    trace_id,
                    span_id=request_time,
                    start_ns=request_time * 1_000_000,
                    end_ns=(request_time + 1_000) * 1_000_000,
                )
            ],
        )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="30d",
            now_millis=now_millis,
        )

    assert archived == 2
    assert store.get_trace_info("tr-now-priority").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-now-retention").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-now-recent").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_id).tags


def test_archive_traces_queries_all_archive_now_groups_before_selecting_bounded_batch(
    store: SqlAlchemyStore,
):
    exp_first = store.create_experiment("archive-now-first-group")
    exp_second = store.create_experiment("archive-now-second-group")
    for experiment_id, older_than in ((exp_first, "1d"), (exp_second, "2d")):
        store.set_experiment_tag(
            experiment_id,
            ExperimentTag(
                TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({"older_than": older_than})
            ),
        )

    first_candidate = sqlalchemy_store_module._TraceArchiveCandidate(
        trace_id="tr-archive-now-newer",
        experiment_id=exp_first,
        timestamp_ms=10,
    )
    second_candidate = sqlalchemy_store_module._TraceArchiveCandidate(
        trace_id="tr-archive-now-older",
        experiment_id=exp_second,
        timestamp_ms=1,
    )
    archived_trace_ids = []

    def record_archive_candidate(*, trace_id, trace_archival_config):
        archived_trace_ids.append(trace_id)
        return True

    with (
        mock.patch.object(
            store,
            "_find_archivable_trace_candidates_for_experiments",
            side_effect=[[first_candidate], [second_candidate]],
        ) as mock_find_candidates,
        mock.patch.object(
            store, "_archive_trace_candidate", side_effect=record_archive_candidate
        ) as mock_archive_candidate,
    ):
        archived = _archive_traces(
            store,
            default_trace_archival_location="file:///unused-archive-root",
            default_retention="30d",
            max_traces=1,
            now_millis=40 * 24 * 60 * 60 * 1000,
        )

    assert archived == 1
    assert mock_find_candidates.call_count == 2
    assert archived_trace_ids == ["tr-archive-now-older"]
    mock_archive_candidate.assert_called_once()


def test_archive_traces_queries_all_regular_groups_before_selecting_bounded_batch(
    store: SqlAlchemyStore,
):
    exp_first = store.create_experiment("archive-regular-first-group")
    exp_second = store.create_experiment("archive-regular-second-group")
    for experiment_id, retention in ((exp_first, "1d"), (exp_second, "2d")):
        store.set_experiment_tag(
            experiment_id,
            ExperimentTag(
                TraceExperimentTagKey.ARCHIVAL_RETENTION,
                json.dumps({"type": "duration", "value": retention}),
            ),
        )

    first_candidate = sqlalchemy_store_module._TraceArchiveCandidate(
        trace_id="tr-regular-newer",
        experiment_id=exp_first,
        timestamp_ms=10,
    )
    second_candidate = sqlalchemy_store_module._TraceArchiveCandidate(
        trace_id="tr-regular-older",
        experiment_id=exp_second,
        timestamp_ms=1,
    )
    archived_trace_ids = []
    queried_experiment_ids = []

    def record_archive_candidate(*, trace_id, trace_archival_config):
        archived_trace_ids.append(trace_id)
        return True

    def find_candidates(*, experiment_ids, session, max_timestamp_millis, limit):
        queried_experiment_ids.append(tuple(experiment_ids))
        if experiment_ids == [exp_first]:
            return [first_candidate]
        if experiment_ids == [exp_second]:
            return [second_candidate]
        return []

    with (
        mock.patch.object(
            store,
            "_find_archivable_trace_candidates_for_experiments",
            side_effect=find_candidates,
        ) as mock_find_candidates,
        mock.patch.object(
            store, "_archive_trace_candidate", side_effect=record_archive_candidate
        ) as mock_archive_candidate,
    ):
        archived = _archive_traces(
            store,
            default_trace_archival_location="file:///unused-archive-root",
            default_retention="30d",
            max_traces=1,
            now_millis=40 * 24 * 60 * 60 * 1000,
        )

    assert archived == 1
    assert (exp_first,) in queried_experiment_ids
    assert (exp_second,) in queried_experiment_ids
    assert mock_find_candidates.call_count >= 2
    assert archived_trace_ids == ["tr-regular-older"]
    mock_archive_candidate.assert_called_once()


def test_archive_traces_respects_workspace_trace_archival_location_overrides(
    store: SqlAlchemyStore, workspaces_enabled: bool
):
    if not workspaces_enabled:
        pytest.skip("Workspace root override behavior only applies when workspaces are enabled.")

    workspace_store = store._get_workspace_provider_instance()
    exp_id = store.create_experiment("archive-workspace-override")
    now_millis = 30 * 24 * 60 * 60 * 1000
    old_request_time = now_millis - 2 * 24 * 60 * 60 * 1000  # 2 days old
    new_request_time = now_millis - 12 * 60 * 60 * 1000  # 12 hours old

    with TempDir() as tmp:
        server_archive_root = Path(tmp.path("server-archive"))
        workspace_artifact_root = Path(tmp.path("workspace-artifacts"))
        workspace_archive_root = Path(tmp.path("workspace-archive"))
        server_archive_root.mkdir()
        workspace_artifact_root.mkdir()
        workspace_archive_root.mkdir()
        workspace_store.update_workspace(
            Workspace(
                name=DEFAULT_WORKSPACE_NAME,
                default_artifact_root=workspace_artifact_root.as_uri(),
                trace_archival_location=workspace_archive_root.as_uri(),
            )
        )

        _create_trace(
            store,
            "tr-workspace-old",
            exp_id,
            request_time=old_request_time,
        )
        _create_trace(
            store,
            "tr-workspace-new",
            exp_id,
            request_time=new_request_time,
        )
        store.log_spans(
            exp_id,
            [
                create_test_span(
                    "tr-workspace-old",
                    span_id=311,
                    start_ns=old_request_time * 1_000_000,
                    end_ns=(old_request_time + 1_000) * 1_000_000,
                )
            ],
        )
        store.log_spans(
            exp_id,
            [
                create_test_span(
                    "tr-workspace-new",
                    span_id=312,
                    start_ns=new_request_time * 1_000_000,
                    end_ns=(new_request_time + 1_000) * 1_000_000,
                )
            ],
        )

        archived = _archive_traces(
            store,
            default_trace_archival_location=server_archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )

        assert archived == 1
        archived_trace_info = store.get_trace_info("tr-workspace-old")
        expected_workspace_archive_uri = append_to_uri_path(
            workspace_archive_root.as_uri(),
            exp_id,
            SqlAlchemyStore.TRACE_FOLDER_NAME,
            "tr-workspace-old",
            SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
        )
        assert archived_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION] == (
            expected_workspace_archive_uri
        )
        assert (
            workspace_archive_root
            / exp_id
            / SqlAlchemyStore.TRACE_FOLDER_NAME
            / "tr-workspace-old"
            / SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
            / "traces.pb"
        ).is_file()
        assert not (
            workspace_artifact_root
            / exp_id
            / SqlAlchemyStore.TRACE_FOLDER_NAME
            / "tr-workspace-old"
            / SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
            / "traces.pb"
        ).exists()
        assert not (
            server_archive_root
            / WORKSPACES_DIR_NAME
            / DEFAULT_WORKSPACE_NAME
            / exp_id
            / SqlAlchemyStore.TRACE_FOLDER_NAME
            / "tr-workspace-old"
            / SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
            / "traces.pb"
        ).exists()

    assert store.get_trace_info("tr-workspace-new").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )


def test_archive_traces_respects_workspace_trace_archival_retention(
    store: SqlAlchemyStore, workspaces_enabled: bool
):
    if not workspaces_enabled:
        pytest.skip("Workspace retention behavior only applies when workspaces are enabled.")

    workspace_store = store._get_workspace_provider_instance()
    workspace_store.update_workspace(
        Workspace(name=DEFAULT_WORKSPACE_NAME, trace_archival_retention="1d")
    )
    exp_id = store.create_experiment("archive-workspace-retention")
    now_millis = 35 * 24 * 60 * 60 * 1000
    old_request_time = now_millis - 2 * 24 * 60 * 60 * 1000  # 2 days old
    new_request_time = now_millis - 12 * 60 * 60 * 1000  # 12 hours old

    _create_trace(store, "tr-workspace-retention-old", exp_id, request_time=old_request_time)
    _create_trace(store, "tr-workspace-retention-new", exp_id, request_time=new_request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                "tr-workspace-retention-old",
                span_id=321,
                start_ns=old_request_time * 1_000_000,
                end_ns=(old_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_id,
        [
            create_test_span(
                "tr-workspace-retention-new",
                span_id=322,
                start_ns=new_request_time * 1_000_000,
                end_ns=(new_request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="30d",
            now_millis=now_millis,
        )

    assert archived == 1
    assert store.get_trace_info("tr-workspace-retention-old").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert store.get_trace_info("tr-workspace-retention-new").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )


def test_archive_traces_respects_workspace_retention_long_retention_allowlist(
    store: SqlAlchemyStore, workspaces_enabled: bool
):
    if not workspaces_enabled:
        pytest.skip("Workspace retention behavior only applies when workspaces are enabled.")

    workspace_store = store._get_workspace_provider_instance()
    workspace_store.update_workspace(
        Workspace(name=DEFAULT_WORKSPACE_NAME, trace_archival_retention="30d")
    )
    exp_allowlisted = store.create_experiment("archive-workspace-allowlisted")
    exp_not_allowlisted = store.create_experiment("archive-workspace-not-allowlisted")
    now_millis = 180 * 24 * 60 * 60 * 1000
    request_time = now_millis - 45 * 24 * 60 * 60 * 1000  # 45 days old

    for exp_id in (exp_allowlisted, exp_not_allowlisted):
        store.set_experiment_tag(
            exp_id,
            ExperimentTag(
                TraceExperimentTagKey.ARCHIVAL_RETENTION,
                json.dumps({"type": "duration", "value": "90d"}),
            ),
        )

    _create_trace(store, "tr-workspace-allowlisted", exp_allowlisted, request_time=request_time)
    _create_trace(
        store,
        "tr-workspace-not-allowlisted",
        exp_not_allowlisted,
        request_time=request_time,
    )
    store.log_spans(
        exp_allowlisted,
        [
            create_test_span(
                "tr-workspace-allowlisted",
                span_id=331,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_not_allowlisted,
        [
            create_test_span(
                "tr-workspace-not-allowlisted",
                span_id=332,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="120d",
            long_retention_allowlist={exp_allowlisted},
            now_millis=now_millis,
        )

    assert archived == 1
    assert store.get_trace_info("tr-workspace-allowlisted").tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )
    assert (
        store.get_trace_info("tr-workspace-not-allowlisted").tags[TraceTagKey.SPANS_LOCATION]
        == SpansLocation.ARCHIVE_REPO.value
    )


def test_archive_traces_noops_when_candidate_becomes_stale(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive-stale-candidate")
    trace_id = "tr-stale-candidate"
    now_millis = 40 * 24 * 60 * 60 * 1000
    _create_trace(store, trace_id, exp_id, request_time=now_millis - 2 * 24 * 60 * 60 * 1000)
    store.log_spans(exp_id, [create_test_span(trace_id, span_id=411)])

    from mlflow.store.artifact.artifact_repo import ArtifactRepository

    original_upload_archived_trace_data_bytes = ArtifactRepository.upload_archived_trace_data_bytes

    def upload_bytes_and_mutate(self, data):
        original_upload_archived_trace_data_bytes(self, data)
        store.log_spans(exp_id, [create_test_span(trace_id, span_id=412)])

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archive_payload_path = (
            archive_root
            / exp_id
            / SqlAlchemyStore.TRACE_FOLDER_NAME
            / trace_id
            / SqlAlchemyStore.ARTIFACTS_FOLDER_NAME
            / "traces.pb"
        )
        with mock.patch.object(
            ArtifactRepository, "upload_archived_trace_data_bytes", new=upload_bytes_and_mutate
        ):
            archived = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="1d",
                now_millis=now_millis,
            )
        assert not archive_payload_path.exists()

    assert archived == 0
    assert store.get_trace_info(trace_id).tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )
    with store.ManagedSessionMaker() as session:
        contents = (
            session
            .query(SqlSpan.content)
            .filter(SqlSpan.trace_id == trace_id)
            .order_by(SqlSpan.span_id.asc())
            .all()
        )
        assert all(content for (content,) in contents)


def test_log_spans_rejects_archived_trace(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive-reject-log-spans")
    trace_id = "tr-archive-reject-log-spans"
    now_millis = 42 * 24 * 60 * 60 * 1000
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=421,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="1d",
            now_millis=now_millis,
        )

        assert archived == 1
        archived_trace_info = store.get_trace_info(trace_id)
        assert (
            archived_trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
        )
        archived_trace = store.get_trace(trace_id)

        with pytest.raises(
            MlflowException, match=f"Cannot log spans to archived traces: '{trace_id}'"
        ) as exc_info:
            store.log_spans(
                exp_id,
                [
                    create_test_span(
                        trace_id,
                        span_id=422,
                        start_ns=(request_time + 2_000) * 1_000_000,
                        end_ns=(request_time + 3_000) * 1_000_000,
                    )
                ],
            )

        assert exc_info.value.error_code == ErrorCode.Name(INVALID_STATE)
        trace_info = store.get_trace_info(trace_id)
        assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
        assert (
            trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]
            == archived_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]
        )
        assert [span.span_id for span in store.get_trace(trace_id).data.spans] == [
            span.span_id for span in archived_trace.data.spans
        ]
        with store.ManagedSessionMaker() as session:
            contents = (
                session
                .query(SqlSpan.content)
                .filter(SqlSpan.trace_id == trace_id)
                .order_by(SqlSpan.span_id.asc())
                .all()
            )
            assert contents == [("",)]


def test_archive_traces_continues_after_upload_failure_and_cleans_up_completed_archive_now_requests(
    store: SqlAlchemyStore,
):
    exp_fail = store.create_experiment("archive-upload-failure")
    exp_success = store.create_experiment("archive-upload-success")
    fail_trace_id = "tr-upload-failure"
    success_trace_id = "tr-upload-success"
    now_millis = 45 * 24 * 60 * 60 * 1000
    fail_request_time = now_millis - 3 * 24 * 60 * 60 * 1000
    success_request_time = now_millis - 2 * 24 * 60 * 60 * 1000

    for exp_id in (exp_fail, exp_success):
        store.set_experiment_tag(
            exp_id,
            ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({})),
        )

    _create_trace(store, fail_trace_id, exp_fail, request_time=fail_request_time)
    _create_trace(store, success_trace_id, exp_success, request_time=success_request_time)
    store.log_spans(
        exp_fail,
        [
            create_test_span(
                fail_trace_id,
                span_id=711,
                start_ns=fail_request_time * 1_000_000,
                end_ns=(fail_request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.log_spans(
        exp_success,
        [
            create_test_span(
                success_trace_id,
                span_id=712,
                start_ns=success_request_time * 1_000_000,
                end_ns=(success_request_time + 1_000) * 1_000_000,
            )
        ],
    )

    from mlflow.store.artifact.artifact_repo import ArtifactRepository

    original_upload_archived_trace_data_bytes = ArtifactRepository.upload_archived_trace_data_bytes
    call_count = [0]

    def upload_bytes_with_failure(self, data):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated archive upload failure")
        return original_upload_archived_trace_data_bytes(self, data)

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        with mock.patch.object(
            ArtifactRepository,
            "upload_archived_trace_data_bytes",
            new=upload_bytes_with_failure,
        ):
            archived = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="365d",
                now_millis=now_millis,
            )

    assert archived == 1
    assert store.get_trace_info(fail_trace_id).tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.TRACKING_STORE.value
    )
    assert store.get_trace_info(success_trace_id).tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert TraceExperimentTagKey.ARCHIVE_NOW in store.get_experiment(exp_fail).tags
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_success).tags


def test_archive_traces_keeps_new_archive_now_request_added_mid_pass(
    store: SqlAlchemyStore, monkeypatch
):
    now_millis = 62 * 24 * 60 * 60 * 1000
    exp_id = store.create_experiment("archive-now-tag-replaced-mid-pass")
    original_request = json.dumps({"older_than": "30d"})
    replacement_request = json.dumps({"older_than": "1h"})
    archived_trace_ids = []

    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, original_request)
    )

    def _find_candidates(*, session, experiment_ids, max_timestamp_millis, limit):
        if exp_id in experiment_ids:
            return [
                _TraceArchiveCandidate(
                    trace_id="tr-replaced-request",
                    experiment_id=exp_id,
                    timestamp_ms=1,
                )
            ]
        return []

    def _archive_candidate(*, trace_id, trace_archival_config):
        archived_trace_ids.append(trace_id)
        store.set_experiment_tag(
            exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, replacement_request)
        )
        return True

    monkeypatch.setattr(
        store,
        "_find_archivable_trace_candidates_for_experiments",
        _find_candidates,
    )
    monkeypatch.setattr(store, "_archive_trace_candidate", _archive_candidate)
    monkeypatch.setattr(store, "_get_archive_now_remaining_state", lambda **_: "done")

    archived = _archive_traces(
        store,
        default_trace_archival_location="file:///unused-archive-root",
        default_retention="365d",
        max_traces=1,
        now_millis=now_millis,
    )

    assert archived == 1
    assert archived_trace_ids == ["tr-replaced-request"]
    assert (
        store.get_experiment(exp_id).tags[TraceExperimentTagKey.ARCHIVE_NOW] == replacement_request
    )


@pytest.mark.parametrize("invalid_content", ["not-json", "[]"], ids=["invalid-json", "wrong-shape"])
def test_archive_traces_marks_malformed_traces_and_excludes_retries(
    store: SqlAlchemyStore, invalid_content: str
):
    exp_id = store.create_experiment("archive-malformed-trace")
    trace_id = "tr-malformed"
    now_millis = 50 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=now_millis - 2 * 24 * 60 * 60 * 1000)
    store.log_spans(exp_id, [create_test_span(trace_id, span_id=511)])
    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )

    with store.ManagedSessionMaker() as session:
        (
            session
            .query(SqlSpan)
            .filter(SqlSpan.trace_id == trace_id)
            .update({SqlSpan.content: invalid_content}, synchronize_session=False)
        )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="365d",
            now_millis=now_millis,
        )
        archived_again = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="365d",
            now_millis=now_millis,
        )

    assert archived == 0
    assert archived_again == 0
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value
    assert trace_info.tags[TraceTagKey.ARCHIVAL_FAILURE] == (
        TraceArchivalFailureReason.MALFORMED_TRACE.value
    )
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_id).tags


def test_archive_traces_marks_serializer_failures_as_malformed_and_excludes_retries(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-serializer-malformed-trace")
    trace_id = "tr-serializer-malformed"
    now_millis = 55 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=now_millis - 2 * 24 * 60 * 60 * 1000)
    store.log_spans(exp_id, [create_test_span(trace_id, span_id=561)])
    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        with mock.patch(
            "mlflow.store.tracking.sqlalchemy_store.spans_to_traces_data_pb",
            side_effect=MlflowException.invalid_parameter_value("simulated malformed trace"),
        ):
            archived = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="365d",
                now_millis=now_millis,
            )
            archived_again = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="365d",
                now_millis=now_millis,
            )

    assert archived == 0
    assert archived_again == 0
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value
    assert trace_info.tags[TraceTagKey.ARCHIVAL_FAILURE] == (
        TraceArchivalFailureReason.MALFORMED_TRACE.value
    )
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_id).tags


def test_archive_traces_marks_unsupported_archive_repository_as_terminal_failure(
    store: SqlAlchemyStore,
):
    exp_fail = store.create_experiment("archive-unsupported-repository")
    exp_success = store.create_experiment("archive-supported-repository")
    fail_trace_id = "tr-unsupported-repository"
    success_trace_id = "tr-supported-repository"
    now_millis = 57 * 24 * 60 * 60 * 1000

    _create_trace(store, fail_trace_id, exp_fail, request_time=now_millis - 2 * 24 * 60 * 60 * 1000)
    _create_trace(
        store, success_trace_id, exp_success, request_time=now_millis - 24 * 60 * 60 * 1000
    )
    store.log_spans(exp_fail, [create_test_span(fail_trace_id, span_id=571)])
    store.log_spans(exp_success, [create_test_span(success_trace_id, span_id=572)])
    for exp_id in (exp_fail, exp_success):
        store.set_experiment_tag(
            exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
        )

    from mlflow.store.artifact.artifact_repo import ArtifactRepository

    original_upload_archived_trace_data_bytes = ArtifactRepository.upload_archived_trace_data_bytes

    def upload_bytes_unsupported(self, data):
        if fail_trace_id in self.artifact_uri:
            raise MlflowNotImplementedException(
                "Databricks trace artifact repositories do not yet support ARCHIVE_REPO trace "
                "payloads."
            )
        return original_upload_archived_trace_data_bytes(self, data)

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        with mock.patch.object(
            ArtifactRepository,
            "upload_archived_trace_data_bytes",
            new=upload_bytes_unsupported,
        ):
            archived = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="365d",
                now_millis=now_millis,
            )

    assert archived == 1
    fail_trace_info = store.get_trace_info(fail_trace_id)
    assert fail_trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value
    assert fail_trace_info.tags[TraceTagKey.ARCHIVAL_FAILURE] == (
        TraceArchivalFailureReason.UNSUPPORTED_ARCHIVE_REPOSITORY.value
    )
    success_trace_info = store.get_trace_info(success_trace_id)
    assert success_trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_fail).tags
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_success).tags


def test_archive_traces_keeps_archive_now_when_only_unmarked_non_archivable_traces_remain(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-now-terminal-non-archivable")
    trace_id = "tr-terminal-non-archivable"
    now_millis = 58 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=now_millis - 2 * 24 * 60 * 60 * 1000)
    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="365d",
            now_millis=now_millis,
        )

    assert archived == 0
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags.get(TraceTagKey.SPANS_LOCATION) is None
    assert TraceTagKey.ARCHIVAL_FAILURE not in trace_info.tags
    assert TraceExperimentTagKey.ARCHIVE_NOW in store.get_experiment(exp_id).tags


def test_archive_traces_raises_unexpected_deserialization_errors(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-unexpected-deserialize-error")
    trace_id = "tr-unexpected-deserialize-error"
    now_millis = 59 * 24 * 60 * 60 * 1000
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=581,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        with mock.patch.object(
            store,
            "_serialize_trace_archival_snapshot_to_pb",
            side_effect=RuntimeError("simulated unexpected serialize failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated unexpected serialize failure"):
                _archive_traces(
                    store,
                    default_trace_archival_location=archive_root.as_uri(),
                    default_retention="365d",
                    now_millis=now_millis,
                )

    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.TRACKING_STORE.value
    assert TraceTagKey.ARCHIVAL_FAILURE not in trace_info.tags
    assert TraceExperimentTagKey.ARCHIVE_NOW in store.get_experiment(exp_id).tags


@pytest.mark.parametrize(
    "error",
    [
        MlflowException("simulated retryable archival failure"),
        OSError("simulated storage error"),
    ],
    ids=["mlflow-error", "os-error"],
)
def test_archive_traces_leaves_retryable_errors_retryable(store: SqlAlchemyStore, error: Exception):
    exp_id = store.create_experiment("archive-retryable-mlflow-error")
    trace_id = "tr-retryable-mlflow-error"
    now_millis = 60 * 24 * 60 * 60 * 1000
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=582,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )

    with TempDir() as tmp:
        from mlflow.store.artifact.artifact_repo import ArtifactRepository

        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        with mock.patch.object(
            ArtifactRepository,
            "upload_archived_trace_data_bytes",
            side_effect=error,
        ):
            archived = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="365d",
                now_millis=now_millis,
            )

        assert archived == 0
        assert TraceExperimentTagKey.ARCHIVE_NOW in store.get_experiment(exp_id).tags

        archived_again = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="365d",
            now_millis=now_millis,
        )

    assert archived_again == 1
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
    assert TraceTagKey.ARCHIVAL_FAILURE not in trace_info.tags
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_id).tags


def test_archive_traces_leaves_sqlalchemy_errors_retryable(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-retryable-sqlalchemy-error")
    trace_id = "tr-retryable-sqlalchemy-error"
    now_millis = 61 * 24 * 60 * 60 * 1000
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000

    _create_trace(store, trace_id, exp_id, request_time=request_time)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id,
                span_id=583,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )
    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        with mock.patch.object(
            store,
            "_load_trace_archival_snapshot",
            side_effect=SQLAlchemyError("simulated retryable db failure"),
        ):
            archived = _archive_traces(
                store,
                default_trace_archival_location=archive_root.as_uri(),
                default_retention="365d",
                now_millis=now_millis,
            )

        assert archived == 0
        assert TraceExperimentTagKey.ARCHIVE_NOW in store.get_experiment(exp_id).tags

        archived_again = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="365d",
            now_millis=now_millis,
        )

    assert archived_again == 1
    trace_info = store.get_trace_info(trace_id)
    assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
    assert TraceTagKey.ARCHIVAL_FAILURE not in trace_info.tags
    assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_id).tags


def test_archive_traces_keeps_archive_now_when_matching_traces_are_transiently_blocked(
    store: SqlAlchemyStore,
):
    exp_id = store.create_experiment("archive-now-transiently-blocked")
    now_millis = 60 * 24 * 60 * 60 * 1000
    archivable_trace_id = "tr-now-archivable"
    in_progress_trace_id = "tr-now-in-progress"
    request_time = now_millis - 2 * 24 * 60 * 60 * 1000  # 2 days old

    store.set_experiment_tag(
        exp_id, ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({}))
    )
    _create_trace(store, archivable_trace_id, exp_id, request_time=request_time)
    _create_trace(
        store,
        in_progress_trace_id,
        exp_id,
        request_time=request_time,
        state=TraceState.IN_PROGRESS,
    )
    store.log_spans(
        exp_id,
        [
            create_test_span(
                archivable_trace_id,
                span_id=611,
                start_ns=request_time * 1_000_000,
                end_ns=(request_time + 1_000) * 1_000_000,
            )
        ],
    )

    with TempDir() as tmp:
        archive_root = Path(tmp.path("archive"))
        archive_root.mkdir()
        archived = _archive_traces(
            store,
            default_trace_archival_location=archive_root.as_uri(),
            default_retention="365d",
            now_millis=now_millis,
        )

    assert archived == 1
    assert store.get_trace_info(archivable_trace_id).tags[TraceTagKey.SPANS_LOCATION] == (
        SpansLocation.ARCHIVE_REPO.value
    )
    assert TraceExperimentTagKey.ARCHIVE_NOW in store.get_experiment(exp_id).tags
