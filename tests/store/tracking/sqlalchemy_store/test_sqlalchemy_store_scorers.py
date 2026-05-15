import json
import math
import time
import uuid
from unittest import mock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
    trace_location,
)
from mlflow.entities.assessment import FeedbackValue
from mlflow.entities.gateway_endpoint import GatewayEndpoint
from mlflow.entities.span import create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlOnlineScoringConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import TraceJSONEncoder

from tests.store.tracking.sqlalchemy_store.conftest import (
    _create_experiments,
    create_mock_span_context,
    create_test_otel_span,
    create_test_span,
)

pytestmark = pytest.mark.notrackingurimock


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_default_trace_status_in_progress(store: SqlAlchemyStore, is_async: bool):
    experiment_id = store.create_experiment("test_default_in_progress")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Create a child span (has parent, not a root span)
    child_context = mock.Mock()
    child_context.trace_id = 56789
    child_context.span_id = 777
    child_context.is_remote = False
    child_context.trace_flags = trace_api.TraceFlags(1)
    child_context.trace_state = trace_api.TraceState()

    parent_context = mock.Mock()
    parent_context.trace_id = 56789
    parent_context.span_id = 888  # Parent span not included in log
    parent_context.is_remote = False
    parent_context.trace_flags = trace_api.TraceFlags(1)
    parent_context.trace_state = trace_api.TraceState()

    child_otel_span = OTelReadableSpan(
        name="child_span_only",
        context=child_context,
        parent=parent_context,  # Has parent, not a root span
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps("LLM", cls=TraceJSONEncoder),
        },
        start_time=2000000000,
        end_time=3000000000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        resource=_OTelResource.get_empty(),
    )
    child_span = create_mlflow_span(child_otel_span, trace_id, "LLM")

    # Log only the child span (no root span)
    if is_async:
        await store.log_spans_async(experiment_id, [child_span])
    else:
        store.log_spans(experiment_id, [child_span])

    # Check trace was created with IN_PROGRESS status (default when no root span)
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "IN_PROGRESS"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("span_status_code", "expected_trace_status"),
    [
        (trace_api.StatusCode.OK, "OK"),
        (trace_api.StatusCode.ERROR, "ERROR"),
    ],
)
async def test_log_spans_sets_trace_status_from_root_span(
    store: SqlAlchemyStore,
    is_async: bool,
    span_status_code: trace_api.StatusCode,
    expected_trace_status: str,
):
    experiment_id = store.create_experiment("test_trace_status_from_root")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Create root span with specified status
    description = (
        f"Root span {span_status_code.name}"
        if span_status_code == trace_api.StatusCode.ERROR
        else None
    )
    root_otel_span = create_test_otel_span(
        trace_id=trace_id,
        name=f"root_span_{span_status_code.name}",
        status_code=span_status_code,
        status_description=description,
        trace_id_num=12345 + span_status_code.value,
        span_id_num=111 + span_status_code.value,
    )
    root_span = create_mlflow_span(root_otel_span, trace_id, "LLM")

    # Log the span
    if is_async:
        await store.log_spans_async(experiment_id, [root_span])
    else:
        store.log_spans(experiment_id, [root_span])

    # Verify trace has expected status from root span
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == expected_trace_status


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_unset_root_span_status_defaults_to_ok(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_unset_root_span")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # Create root span with UNSET status (this is unexpected in practice)
    root_unset_span = create_test_otel_span(
        trace_id=trace_id,
        name="root_span_unset",
        status_code=trace_api.StatusCode.UNSET,  # Unexpected in practice
        start_time=3000000000,
        end_time=4000000000,
        trace_id_num=23456,
        span_id_num=333,
    )
    root_span = create_mlflow_span(root_unset_span, trace_id, "LLM")

    if is_async:
        await store.log_spans_async(experiment_id, [root_span])
    else:
        store.log_spans(experiment_id, [root_span])

    # Verify trace defaults to OK status when root span has UNSET status
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_in_progress_trace_status_from_root_span(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_trace_status_update")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # First, log a non-root span which will create trace with default IN_PROGRESS status
    parent_context = create_mock_span_context(45678, 555)  # Will be root span later

    child_otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="child_span",
        parent=parent_context,  # Has parent, not a root span
        status_code=trace_api.StatusCode.OK,
        start_time=1100000000,
        end_time=1900000000,
        trace_id_num=45678,
        span_id_num=666,
    )
    child_span = create_mlflow_span(child_otel_span, trace_id, "LLM")

    if is_async:
        await store.log_spans_async(experiment_id, [child_span])
    else:
        store.log_spans(experiment_id, [child_span])

    # Verify trace was created with IN_PROGRESS status (default when no root span)
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "IN_PROGRESS"

    # Now log root span with ERROR status
    root_otel_span = create_test_otel_span(
        trace_id=trace_id,
        name="root_span",
        parent=None,  # Root span
        status_code=trace_api.StatusCode.ERROR,
        status_description="Root span error",
        trace_id_num=45678,
        span_id_num=555,
    )
    root_span = create_mlflow_span(root_otel_span, trace_id, "LLM")

    if is_async:
        await store.log_spans_async(experiment_id, [root_span])
    else:
        store.log_spans(experiment_id, [root_span])

    # Check trace status was updated to ERROR from root span
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_updates_state_unspecified_trace_status_from_root_span(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_unspecified_update")
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id = f"tr-{uuid.uuid4().hex}"

    # First, create a trace with OK status by logging a root span with OK status
    initial_span = create_test_span(
        trace_id=trace_id,
        name="initial_unset_span",
        span_id=999,
        status=trace_api.StatusCode.OK,
        trace_num=67890,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [initial_span])
    else:
        store.log_spans(experiment_id, [initial_span])

    # Verify trace was created with OK status
    trace = store.get_trace_info(trace_id)
    assert trace.state.value == "OK"

    # Now log a new root span with OK status (earlier start time makes it the new root)
    new_root_span = create_test_span(
        trace_id=trace_id,
        name="new_root_span",
        span_id=1000,
        status=trace_api.StatusCode.OK,
        start_ns=500000000,  # Earlier than initial span
        end_ns=2500000000,
        trace_num=67890,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [new_root_span])
    else:
        store.log_spans(experiment_id, [new_root_span])

    # Check trace status was updated to OK from root span
    traces, _ = store.search_traces([experiment_id])
    trace = next(t for t in traces if t.request_id == trace_id)
    assert trace.state.value == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_log_spans_does_not_update_finalized_trace_status(
    store: SqlAlchemyStore, is_async: bool
):
    experiment_id = store.create_experiment("test_no_update_finalized")

    # Test that OK status is not updated
    # Generate a proper MLflow trace ID in the format "tr-<32-char-hex>"
    trace_id_ok = f"tr-{uuid.uuid4().hex}"

    # Create initial root span with OK status
    ok_span = create_test_span(
        trace_id=trace_id_ok,
        name="ok_root_span",
        span_id=1111,
        status=trace_api.StatusCode.OK,
        trace_num=78901,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [ok_span])
    else:
        store.log_spans(experiment_id, [ok_span])

    # Verify trace has OK status
    traces, _ = store.search_traces([experiment_id])
    trace_ok = next(t for t in traces if t.request_id == trace_id_ok)
    assert trace_ok.state.value == "OK"

    # Now log a new root span with ERROR status
    error_span = create_test_span(
        trace_id=trace_id_ok,
        name="error_root_span",
        span_id=2222,
        status=trace_api.StatusCode.ERROR,
        status_desc="New error",
        start_ns=500000000,
        end_ns=2500000000,
        trace_num=78901,
    )

    if is_async:
        await store.log_spans_async(experiment_id, [error_span])
    else:
        store.log_spans(experiment_id, [error_span])

    # Verify trace status is still OK (not updated to ERROR)
    traces, _ = store.search_traces([experiment_id])
    trace_ok = next(t for t in traces if t.request_id == trace_id_ok)
    assert trace_ok.state.value == "OK"


def test_scorer_operations(store: SqlAlchemyStore):
    """
    Test the scorer operations: register_scorer, list_scorers, get_scorer, and delete_scorer.

    This test covers:
    1. Registering multiple scorers with different names
    2. Registering multiple versions of the same scorer
    3. Listing scorers (should return latest version for each name)
    4. Getting specific scorer versions
    5. Getting latest scorer version when version is not specified
    6. Deleting scorers and verifying they are deleted
    """
    # Create an experiment for testing
    experiment_id = store.create_experiment("test_scorer_experiment")

    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer1"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer2"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer3"}')

    store.register_scorer(experiment_id, "safety_scorer", '{"data": "safety_scorer1"}')
    store.register_scorer(experiment_id, "safety_scorer", '{"data": "safety_scorer2"}')

    store.register_scorer(experiment_id, "relevance_scorer", '{"data": "relevance_scorer1"}')

    # Step 2: Test list_scorers - should return latest version for each scorer name
    scorers = store.list_scorers(experiment_id)

    # Should return 3 scorers (one for each unique name)
    assert len(scorers) == 3, f"Expected 3 scorers, got {len(scorers)}"

    scorer_names = [scorer.scorer_name for scorer in scorers]
    # Verify the order is sorted by scorer_name
    assert scorer_names == ["accuracy_scorer", "relevance_scorer", "safety_scorer"], (
        f"Expected sorted order, got {scorer_names}"
    )

    # Verify versions are the latest and check serialized_scorer content
    for scorer in scorers:
        if scorer.scorer_name == "accuracy_scorer":
            assert scorer.scorer_version == 3, (
                f"Expected version 3 for accuracy_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == '{"data": "accuracy_scorer3"}'
        elif scorer.scorer_name == "safety_scorer":
            assert scorer.scorer_version == 2, (
                f"Expected version 2 for safety_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == '{"data": "safety_scorer2"}'
        elif scorer.scorer_name == "relevance_scorer":
            assert scorer.scorer_version == 1, (
                f"Expected version 1 for relevance_scorer, got {scorer.scorer_version}"
            )
            assert scorer._serialized_scorer == '{"data": "relevance_scorer1"}'

    # Test list_scorer_versions
    accuracy_scorer_versions = store.list_scorer_versions(experiment_id, "accuracy_scorer")
    assert len(accuracy_scorer_versions) == 3, (
        f"Expected 3 versions, got {len(accuracy_scorer_versions)}"
    )

    # Verify versions are ordered by version number
    assert accuracy_scorer_versions[0].scorer_version == 1
    assert accuracy_scorer_versions[0]._serialized_scorer == '{"data": "accuracy_scorer1"}'
    assert accuracy_scorer_versions[1].scorer_version == 2
    assert accuracy_scorer_versions[1]._serialized_scorer == '{"data": "accuracy_scorer2"}'
    assert accuracy_scorer_versions[2].scorer_version == 3
    assert accuracy_scorer_versions[2]._serialized_scorer == '{"data": "accuracy_scorer3"}'

    # Step 3: Test get_scorer with specific versions
    # Get accuracy_scorer version 1
    accuracy_v1 = store.get_scorer(experiment_id, "accuracy_scorer", version=1)
    assert accuracy_v1._serialized_scorer == '{"data": "accuracy_scorer1"}'
    assert accuracy_v1.scorer_version == 1

    # Get accuracy_scorer version 2
    accuracy_v2 = store.get_scorer(experiment_id, "accuracy_scorer", version=2)
    assert accuracy_v2._serialized_scorer == '{"data": "accuracy_scorer2"}'
    assert accuracy_v2.scorer_version == 2

    # Get accuracy_scorer version 3 (latest)
    accuracy_v3 = store.get_scorer(experiment_id, "accuracy_scorer", version=3)
    assert accuracy_v3._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_v3.scorer_version == 3

    # Step 4: Test get_scorer without version (should return latest)
    accuracy_latest = store.get_scorer(experiment_id, "accuracy_scorer")
    assert accuracy_latest._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_latest.scorer_version == 3

    safety_latest = store.get_scorer(experiment_id, "safety_scorer")
    assert safety_latest._serialized_scorer == '{"data": "safety_scorer2"}'
    assert safety_latest.scorer_version == 2

    relevance_latest = store.get_scorer(experiment_id, "relevance_scorer")
    assert relevance_latest._serialized_scorer == '{"data": "relevance_scorer1"}'
    assert relevance_latest.scorer_version == 1

    # Step 5: Test error cases for get_scorer
    # Try to get non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent' not found"):
        store.get_scorer(experiment_id, "non_existent")

    # Try to get non-existent version
    with pytest.raises(
        MlflowException, match="Scorer with name 'accuracy_scorer' and version 999 not found"
    ):
        store.get_scorer(experiment_id, "accuracy_scorer", version=999)

    # Step 6: Test delete_scorer - delete specific version of accuracy_scorer
    # Delete version 1 of accuracy_scorer
    store.delete_scorer(experiment_id, "accuracy_scorer", version=1)

    # Verify version 1 is deleted but other versions still exist
    with pytest.raises(
        MlflowException, match="Scorer with name 'accuracy_scorer' and version 1 not found"
    ):
        store.get_scorer(experiment_id, "accuracy_scorer", version=1)

    # Verify versions 2 and 3 still exist
    accuracy_v2 = store.get_scorer(experiment_id, "accuracy_scorer", version=2)
    assert accuracy_v2._serialized_scorer == '{"data": "accuracy_scorer2"}'
    assert accuracy_v2.scorer_version == 2

    accuracy_v3 = store.get_scorer(experiment_id, "accuracy_scorer", version=3)
    assert accuracy_v3._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_v3.scorer_version == 3

    # Verify latest version still works
    accuracy_latest_after_partial_delete = store.get_scorer(experiment_id, "accuracy_scorer")
    assert accuracy_latest_after_partial_delete._serialized_scorer == '{"data": "accuracy_scorer3"}'
    assert accuracy_latest_after_partial_delete.scorer_version == 3

    # Step 7: Test delete_scorer - delete all versions of accuracy_scorer
    store.delete_scorer(experiment_id, "accuracy_scorer")

    # Verify accuracy_scorer is completely deleted
    with pytest.raises(MlflowException, match="Scorer with name 'accuracy_scorer' not found"):
        store.get_scorer(experiment_id, "accuracy_scorer")

    # Verify other scorers still exist
    safety_latest_after_delete = store.get_scorer(experiment_id, "safety_scorer")
    assert safety_latest_after_delete._serialized_scorer == '{"data": "safety_scorer2"}'
    assert safety_latest_after_delete.scorer_version == 2

    relevance_latest_after_delete = store.get_scorer(experiment_id, "relevance_scorer")
    assert relevance_latest_after_delete._serialized_scorer == '{"data": "relevance_scorer1"}'
    assert relevance_latest_after_delete.scorer_version == 1

    # Step 8: Test list_scorers after deletion
    scorers_after_delete = store.list_scorers(experiment_id)
    assert len(scorers_after_delete) == 2, (
        f"Expected 2 scorers after deletion, got {len(scorers_after_delete)}"
    )

    scorer_names_after_delete = [scorer.scorer_name for scorer in scorers_after_delete]
    assert "accuracy_scorer" not in scorer_names_after_delete
    assert "safety_scorer" in scorer_names_after_delete
    assert "relevance_scorer" in scorer_names_after_delete

    # Step 9: Test delete_scorer for non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent' not found"):
        store.delete_scorer(experiment_id, "non_existent")

    # Step 10: Test delete_scorer for non-existent version
    with pytest.raises(
        MlflowException, match="Scorer with name 'safety_scorer' and version 999 not found"
    ):
        store.delete_scorer(experiment_id, "safety_scorer", version=999)

    # Step 11: Test delete_scorer for remaining scorers
    store.delete_scorer(experiment_id, "safety_scorer")
    store.delete_scorer(experiment_id, "relevance_scorer")

    # Verify all scorers are deleted
    final_scorers = store.list_scorers(experiment_id)
    assert len(final_scorers) == 0, (
        f"Expected 0 scorers after all deletions, got {len(final_scorers)}"
    )

    # Step 12: Test list_scorer_versions
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer1"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer2"}')
    store.register_scorer(experiment_id, "accuracy_scorer", '{"data": "accuracy_scorer3"}')

    # Test list_scorer_versions for non-existent scorer
    with pytest.raises(MlflowException, match="Scorer with name 'non_existent_scorer' not found"):
        store.list_scorer_versions(experiment_id, "non_existent_scorer")


@pytest.mark.parametrize(
    ("name", "error_match"),
    [
        (None, "cannot be None"),
        (123, "must be a string"),
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
    ],
)
def test_register_scorer_validates_name(store: SqlAlchemyStore, name, error_match):
    experiment_id = store.create_experiment("test_scorer_name_validation")
    with pytest.raises(MlflowException, match=error_match):
        store.register_scorer(experiment_id, name, '{"data": "test"}')


@pytest.mark.parametrize(
    ("model", "error_match"),
    [
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
    ],
)
def test_register_scorer_validates_model(store: SqlAlchemyStore, model, error_match):
    experiment_id = store.create_experiment("test_scorer_model_validation")
    scorer_json = json.dumps({"instructions_judge_pydantic_data": {"model": model}})
    with pytest.raises(MlflowException, match=error_match):
        store.register_scorer(experiment_id, "test_scorer", scorer_json)


def _gateway_model_scorer_json():
    return json.dumps({"instructions_judge_pydantic_data": {"model": "gateway:/my-endpoint"}})


def _non_gateway_model_scorer_json():
    return json.dumps({"instructions_judge_pydantic_data": {"model": "openai:/gpt-4"}})


def _mock_gateway_endpoint():
    return GatewayEndpoint(
        endpoint_id="test-endpoint-id",
        name="my-endpoint",
        created_at=0,
        last_updated_at=0,
    )


def test_get_online_scoring_configs_batch(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_batch_configs")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer1", _gateway_model_scorer_json())
        store.register_scorer(experiment_id, "scorer2", _gateway_model_scorer_json())
        store.register_scorer(experiment_id, "scorer3", _gateway_model_scorer_json())

    config1 = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer1",
        sample_rate=0.1,
        filter_string="status = 'OK'",
    )
    config2 = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer2",
        sample_rate=0.5,
    )

    scorer_ids = [config1.scorer_id, config2.scorer_id]
    configs = store.get_online_scoring_configs(scorer_ids)

    assert len(configs) == 2
    configs_by_id = {c.scorer_id: c for c in configs}
    assert configs_by_id[config1.scorer_id].sample_rate == 0.1
    assert configs_by_id[config1.scorer_id].filter_string == "status = 'OK'"
    assert configs_by_id[config2.scorer_id].sample_rate == 0.5
    assert configs_by_id[config2.scorer_id].filter_string is None


def test_get_online_scoring_configs_empty_list(store: SqlAlchemyStore):
    configs = store.get_online_scoring_configs([])
    assert configs == []


def test_get_online_scoring_configs_nonexistent_ids(store: SqlAlchemyStore):
    configs = store.get_online_scoring_configs(["nonexistent_id_1", "nonexistent_id_2"])
    assert configs == []


def test_upsert_online_scoring_config_creates_config(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_create")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.1,
        filter_string="status = 'OK'",
    )

    assert config.sample_rate == 0.1
    assert config.filter_string == "status = 'OK'"
    assert config.online_scoring_config_id is not None


def test_upsert_online_scoring_config_overwrites(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_overwrite")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.1,
    )

    new_config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
    )

    assert new_config.sample_rate == 0.5

    # Verify the config is persisted by fetching via get_online_scoring_configs
    configs = store.get_online_scoring_configs([new_config.scorer_id])
    assert len(configs) == 1
    assert configs[0].scorer_id == new_config.scorer_id
    assert configs[0].sample_rate == 0.5


def test_upsert_online_scoring_config_rejects_non_gateway_model(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_non_gateway")
    non_gateway_scorer = json.dumps({
        "instructions_judge_pydantic_data": {"model": "openai:/gpt-4"}
    })
    store.register_scorer(experiment_id, "scorer", non_gateway_scorer)

    with pytest.raises(MlflowException, match="does not use a gateway model"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="scorer",
            sample_rate=0.1,
        )


def test_upsert_online_scoring_config_rejects_scorer_requiring_expectations(
    store: SqlAlchemyStore,
):
    experiment_id = store.create_experiment("test_online_config_expectations")

    # Complete serialized scorer with {{ expectations }} template variable
    expectations_scorer = json.dumps({
        "name": "expectations_scorer",
        "description": None,
        "aggregations": [],
        "is_session_level_scorer": False,
        "mlflow_version": "3.0.0",
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {
            "model": "gateway:/my-endpoint",
            "instructions": "Compare {{ outputs }} against {{ expectations }}",
        },
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
    })

    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "expectations_scorer", expectations_scorer)

    # Mock LiteLLM availability to allow scorer deserialization during validation
    with mock.patch("mlflow.genai.judges.utils._is_litellm_available", return_value=True):
        with pytest.raises(MlflowException, match="requires expectations.*not currently supported"):
            store.upsert_online_scoring_config(
                experiment_id=experiment_id,
                scorer_name="expectations_scorer",
                sample_rate=0.1,
            )

        # Setting sample_rate to 0 should work (disables automatic evaluation)
        config = store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="expectations_scorer",
            sample_rate=0.0,
        )
        assert config.sample_rate == 0.0


def test_upsert_online_scoring_config_nonexistent_scorer(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_online_config_error")

    with pytest.raises(MlflowException, match="not found"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="nonexistent",
            sample_rate=0.1,
        )


def test_upsert_online_scoring_config_validates_filter_string(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_filter_validation")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "test_scorer", _gateway_model_scorer_json())

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="test_scorer",
        sample_rate=0.5,
        filter_string="status = 'OK'",
    )
    assert config.filter_string == "status = 'OK'"

    with pytest.raises(MlflowException, match="Invalid clause"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate=0.5,
            filter_string="this is not a valid filter !!@@##",
        )


def test_upsert_online_scoring_config_validates_sample_rate(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_sample_rate_validation")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "test_scorer", _gateway_model_scorer_json())

    # Valid sample rates should work
    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="test_scorer",
        sample_rate=0.0,
    )
    assert config.sample_rate == 0.0

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="test_scorer",
        sample_rate=1.0,
    )
    assert config.sample_rate == 1.0

    # Invalid sample rates should raise
    with pytest.raises(MlflowException, match="sample_rate must be between 0.0 and 1.0"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate=-0.1,
        )

    # Non-numeric sample_rate should raise
    with pytest.raises(MlflowException, match="sample_rate must be a number"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate="0.5",
        )

    # Non-string filter_string should raise
    with pytest.raises(MlflowException, match="filter_string must be a string"):
        store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name="test_scorer",
            sample_rate=0.5,
            filter_string=123,
        )


def test_get_active_online_scorers_filters_by_sample_rate(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_active_configs")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "active", _gateway_model_scorer_json())
        store.register_scorer(experiment_id, "inactive", _gateway_model_scorer_json())

    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="active",
        sample_rate=0.1,
    )
    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="inactive",
        sample_rate=0.0,
    )

    active_scorers = store.get_active_online_scorers()
    # Filter to only scorers we created in this test using name and experiment_id
    test_scorers = [
        s
        for s in active_scorers
        if s.name == "active" and s.online_config.experiment_id == experiment_id
    ]

    assert len(test_scorers) == 1
    assert test_scorers[0].online_config.sample_rate == 0.1


def test_get_active_online_scorers_returns_scorer_fields(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_active_configs_info")
    scorer_json = _gateway_model_scorer_json()
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", scorer_json)

    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
        filter_string="status = 'OK'",
    )

    active_scorers = store.get_active_online_scorers()
    active_scorer = next(
        s
        for s in active_scorers
        if s.name == "scorer" and s.online_config.experiment_id == experiment_id
    )

    assert active_scorer.name == "scorer"
    assert active_scorer.online_config.experiment_id == experiment_id
    assert active_scorer.online_config.sample_rate == 0.5
    assert active_scorer.online_config.filter_string == "status = 'OK'"


def test_get_active_online_scorers_filters_non_gateway_model(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_filter_non_gateway")

    # Register scorer with gateway model (version 1)
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    # Set up online scoring config (validation passes for version 1)
    store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
    )

    # Verify scorer is returned initially (max version uses gateway model)
    active_scorers = store.get_active_online_scorers()
    test_scorers = [
        s
        for s in active_scorers
        if s.name == "scorer" and s.online_config.experiment_id == experiment_id
    ]
    assert len(test_scorers) == 1

    # Register same scorer with non-gateway model (version 2)
    store.register_scorer(experiment_id, "scorer", _non_gateway_model_scorer_json())

    # Verify scorer is NOT returned now (max version uses non-gateway model)
    active_scorers = store.get_active_online_scorers()
    test_scorers = [
        s
        for s in active_scorers
        if s.name == "scorer" and s.online_config.experiment_id == experiment_id
    ]
    assert len(test_scorers) == 0


def test_scorer_deletion_cascades_to_online_configs(store: SqlAlchemyStore):
    experiment_id = store.create_experiment("test_cascade_delete")
    with mock.patch.object(store, "get_gateway_endpoint", return_value=_mock_gateway_endpoint()):
        store.register_scorer(experiment_id, "scorer", _gateway_model_scorer_json())

    config = store.upsert_online_scoring_config(
        experiment_id=experiment_id,
        scorer_name="scorer",
        sample_rate=0.5,
    )
    config_id = config.online_scoring_config_id

    with store.ManagedSessionMaker() as session:
        assert (
            session
            .query(SqlOnlineScoringConfig)
            .filter_by(online_scoring_config_id=config_id)
            .count()
            == 1
        )

    store.delete_scorer(experiment_id, "scorer")

    with store.ManagedSessionMaker() as session:
        assert (
            session
            .query(SqlOnlineScoringConfig)
            .filter_by(online_scoring_config_id=config_id)
            .count()
            == 0
        )


def test_dataset_experiment_associations(store):
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        exp_ids = _create_experiments(
            store, ["exp_assoc_1", "exp_assoc_2", "exp_assoc_3", "exp_assoc_4"]
        )
        exp1, exp2, exp3, exp4 = exp_ids

        dataset = store.create_dataset(
            name="test_dataset_associations", experiment_ids=[exp1], tags={"test": "associations"}
        )

        assert dataset.experiment_ids == [exp1]

        updated = store.add_dataset_to_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp3]
        )
        assert set(updated.experiment_ids) == {exp1, exp2, exp3}

        result = store.add_dataset_to_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp4]
        )
        assert set(result.experiment_ids) == {exp1, exp2, exp3, exp4}

        removed = store.remove_dataset_from_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp3]
        )
        assert set(removed.experiment_ids) == {exp1, exp4}

        with mock.patch("mlflow.store.tracking.sqlalchemy_store._logger.warning") as mock_warning:
            idempotent = store.remove_dataset_from_experiments(
                dataset_id=dataset.dataset_id, experiment_ids=[exp2, exp3]
            )
            assert mock_warning.call_count == 2
            assert "was not associated" in mock_warning.call_args_list[0][0][0]

        assert set(idempotent.experiment_ids) == {exp1, exp4}

        with pytest.raises(MlflowException, match="not found"):
            store.add_dataset_to_experiments(dataset_id="d-nonexistent", experiment_ids=[exp1])

        with pytest.raises(MlflowException, match=r"No Experiment with id="):
            store.add_dataset_to_experiments(
                dataset_id=dataset.dataset_id, experiment_ids=["999999"]
            )

        with pytest.raises(MlflowException, match="not found"):
            store.remove_dataset_from_experiments(dataset_id="d-nonexistent", experiment_ids=[exp1])


def _create_simple_trace(store, experiment_id, tags=None):
    trace_id = f"tr-{uuid.uuid4()}"
    timestamp_ms = time.time_ns() // 1_000_000

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=timestamp_ms,
        execution_duration=100,
        state=TraceState.OK,
        tags=tags or {},
    )

    return store.start_trace(trace_info)


def _create_trace_for_correlation(store, experiment_id, spans=None, assessments=None, tags=None):
    trace_id = f"tr-{uuid.uuid4()}"
    timestamp_ms = time.time_ns() // 1_000_000

    trace_tags = tags or {}

    if spans:
        span_types = [span.get("type", "LLM") for span in spans]
        span_statuses = [span.get("status", "OK") for span in spans]

        if "TOOL" in span_types:
            trace_tags["primary_span_type"] = "TOOL"
        elif "LLM" in span_types:
            trace_tags["primary_span_type"] = "LLM"

        if "LLM" in span_types:
            trace_tags["has_llm"] = "true"
        if "TOOL" in span_types:
            trace_tags["has_tool"] = "true"

        trace_tags["has_error"] = "true" if "ERROR" in span_statuses else "false"

        tool_count = sum(1 for t in span_types if t == "TOOL")
        if tool_count > 0:
            trace_tags["tool_count"] = str(tool_count)

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=timestamp_ms,
        execution_duration=100,
        state=TraceState.OK,
        tags=trace_tags,
    )
    store.start_trace(trace_info)

    if assessments:
        for assessment_data in assessments:
            assessment = Feedback(
                assessment_id=assessment_data.get("assessment_id", f"fb-{uuid.uuid4()}"),
                trace_id=trace_id,
                name=assessment_data.get("name", "quality"),
                assessment_type=assessment_data.get("assessment_type", "feedback"),
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id=assessment_data.get("source_id", "user123"),
                ),
                value=FeedbackValue(assessment_data.get("value", 0.8)),
                created_timestamp=timestamp_ms,
                last_updated_timestamp=timestamp_ms,
            )
            store.log_assessments([assessment])

    return trace_id


def _create_trace_with_spans_for_correlation(store, experiment_id, span_configs):
    return _create_trace_for_correlation(store, experiment_id, spans=span_configs)


def test_calculate_trace_filter_correlation_basic(store):
    exp_id = _create_experiments(store, "correlation_test")

    for i in range(10):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "tool_operation", "type": "TOOL", "status": "ERROR"}],
        )

    for i in range(5):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "llm_call", "type": "LLM", "status": "OK"}],
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.primary_span_type = "TOOL"',
        filter_string2='tags.has_error = "true"',
    )

    assert result.npmi == pytest.approx(1.0)
    assert result.filter1_count == 10
    assert result.filter2_count == 10
    assert result.joint_count == 10
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_perfect(store):
    exp_id = _create_experiments(store, "correlation_test")

    for i in range(8):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "operation", "type": "TOOL", "status": "ERROR"}],
        )

    for i in range(7):
        _create_trace_with_spans_for_correlation(
            store,
            exp_id,
            span_configs=[{"name": "operation", "type": "LLM", "status": "OK"}],
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.primary_span_type = "TOOL"',
        filter_string2='tags.has_error = "true"',
    )

    assert result.npmi == pytest.approx(1.0)
    assert result.npmi_smoothed > 0.8
    assert result.filter1_count == 8
    assert result.filter2_count == 8
    assert result.joint_count == 8
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_count_expressions(store):
    exp_id = _create_experiments(store, "correlation_test")

    for i in range(15):
        num_tool_calls = 5 if i < 10 else 2
        spans = [{"type": "TOOL", "name": f"tool_{j}"} for j in range(num_tool_calls)]
        spans.append({"type": "LLM", "name": "llm_call"})
        _create_trace_with_spans_for_correlation(store, exp_id, span_configs=spans)

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.tool_count = "5"',
        filter_string2='tags.has_llm = "true"',
    )

    assert result.filter1_count == 10
    assert result.filter2_count == 15
    assert result.joint_count == 10
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_negative_correlation(store):
    exp_id = _create_experiments(store, "negative_correlation_test")

    for i in range(10):
        _create_trace_for_correlation(
            store, exp_id, spans=[{"type": "LLM", "status": "ERROR"}], tags={"version": "v1"}
        )

    for i in range(10):
        _create_trace_for_correlation(
            store, exp_id, spans=[{"type": "LLM", "status": "OK"}], tags={"version": "v2"}
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.version = "v1"',
        filter_string2='tags.has_error = "false"',
    )

    assert result.total_count == 20
    assert result.filter1_count == 10
    assert result.filter2_count == 10
    assert result.joint_count == 0
    assert result.npmi == pytest.approx(-1.0)


def test_calculate_trace_filter_correlation_zero_counts(store):
    exp_id = _create_experiments(store, "zero_counts_test")

    for i in range(5):
        _create_trace_for_correlation(store, exp_id, spans=[{"type": "LLM", "status": "OK"}])

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.has_llm = "true"',
    )

    assert result.total_count == 5
    assert result.filter1_count == 0
    assert result.filter2_count == 5
    assert result.joint_count == 0
    assert math.isnan(result.npmi)


def test_calculate_trace_filter_correlation_multiple_experiments(store):
    exp_id1 = _create_experiments(store, "multi_exp_1")
    exp_id2 = _create_experiments(store, "multi_exp_2")

    for i in range(4):
        _create_trace_for_correlation(
            store, exp_id1, spans=[{"type": "TOOL", "status": "OK"}], tags={"env": "prod"}
        )

    _create_trace_for_correlation(
        store, exp_id1, spans=[{"type": "LLM", "status": "OK"}], tags={"env": "prod"}
    )

    _create_trace_for_correlation(
        store, exp_id2, spans=[{"type": "TOOL", "status": "OK"}], tags={"env": "dev"}
    )

    for i in range(4):
        _create_trace_for_correlation(
            store, exp_id2, spans=[{"type": "LLM", "status": "OK"}], tags={"env": "dev"}
        )

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id1, exp_id2],
        filter_string1='tags.env = "prod"',
        filter_string2='tags.primary_span_type = "TOOL"',
    )

    assert result.total_count == 10
    assert result.filter1_count == 5
    assert result.filter2_count == 5
    assert result.joint_count == 4
    assert result.npmi > 0.4


def test_calculate_trace_filter_correlation_independent_events(store):
    exp_id = _create_experiments(store, "independent_test")

    configurations = [
        *[{"spans": [{"type": "TOOL", "status": "ERROR"}]} for _ in range(5)],
        *[{"spans": [{"type": "TOOL", "status": "OK"}]} for _ in range(5)],
        *[{"spans": [{"type": "LLM", "status": "ERROR"}]} for _ in range(5)],
        *[{"spans": [{"type": "LLM", "status": "OK"}]} for _ in range(5)],
    ]

    for config in configurations:
        _create_trace_for_correlation(store, exp_id, **config)

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.primary_span_type = "TOOL"',
        filter_string2='tags.has_error = "true"',
    )

    assert result.total_count == 20
    assert result.filter1_count == 10
    assert result.filter2_count == 10
    assert result.joint_count == 5

    # Independent events should have NPMI close to 0
    # P(TOOL) = 10/20 = 0.5, P(ERROR) = 10/20 = 0.5
    # P(TOOL & ERROR) = 5/20 = 0.25
    # Expected joint = 0.5 * 0.5 * 20 = 5, so no correlation
    assert abs(result.npmi) < 0.1


def test_calculate_trace_filter_correlation_simplified_example(store):
    exp_id = _create_experiments(store, "simple_correlation_test")

    for _ in range(5):
        _create_simple_trace(store, exp_id, {"category": "A", "status": "success"})

    for _ in range(3):
        _create_simple_trace(store, exp_id, {"category": "A", "status": "failure"})

    for _ in range(7):
        _create_simple_trace(store, exp_id, {"category": "B", "status": "success"})

    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.category = "A"',
        filter_string2='tags.status = "success"',
    )

    assert result.filter1_count == 8
    assert result.filter2_count == 12
    assert result.joint_count == 5
    assert result.total_count == 15


def test_calculate_trace_filter_correlation_empty_experiment_list(store):
    result = store.calculate_trace_filter_correlation(
        experiment_ids=[],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.primary_span_type = "TOOL"',
    )

    assert result.total_count == 0
    assert result.filter1_count == 0
    assert result.filter2_count == 0
    assert result.joint_count == 0
    assert math.isnan(result.npmi)


def test_calculate_trace_filter_correlation_with_base_filter(store):
    exp_id = _create_experiments(store, "base_filter_test")

    early_time = 1000000000000
    for i in range(5):
        trace_info = TraceInfo(
            trace_id=f"tr-early-{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=early_time + i,
            execution_duration=100,
            state=TraceState.OK,
            tags={
                "has_error": "true" if i < 3 else "false",
                "has_tool": "true" if i % 2 == 0 else "false",
            },
        )
        store.start_trace(trace_info)

    later_time = 2000000000000
    # Create traces in the later period:
    # - 10 total traces in the time window
    # - 6 with has_error=true
    # - 4 with has_tool=true
    # - 3 with both has_error=true AND has_tool=true
    for i in range(10):
        tags = {}
        if i < 6:
            tags["has_error"] = "true"
        if i < 3 or i == 6:
            tags["has_tool"] = "true"

        trace_info = TraceInfo(
            trace_id=f"tr-later-{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=later_time + i,
            execution_duration=100,
            state=TraceState.OK,
            tags=tags,
        )
        store.start_trace(trace_info)

    base_filter = f"timestamp_ms >= {later_time} and timestamp_ms < {later_time + 100}"
    result = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.has_tool = "true"',
        base_filter=base_filter,
    )

    assert result.total_count == 10
    assert result.filter1_count == 6
    assert result.filter2_count == 4
    assert result.joint_count == 3

    # Calculate expected NPMI
    # P(error) = 6/10 = 0.6
    # P(tool) = 4/10 = 0.4
    # P(error AND tool) = 3/10 = 0.3
    # PMI = log(P(error AND tool) / (P(error) * P(tool))) = log(0.3 / (0.6 * 0.4)) = log(1.25)
    # NPMI = PMI / -log(P(error AND tool)) = log(1.25) / -log(0.3)

    p_error = 6 / 10
    p_tool = 4 / 10
    p_joint = 3 / 10

    if p_joint > 0:
        pmi = math.log(p_joint / (p_error * p_tool))
        npmi = pmi / -math.log(p_joint)
        assert abs(result.npmi - npmi) < 0.001

    result_no_base = store.calculate_trace_filter_correlation(
        experiment_ids=[exp_id],
        filter_string1='tags.has_error = "true"',
        filter_string2='tags.has_tool = "true"',
    )

    assert result_no_base.total_count == 15
    assert result_no_base.filter1_count == 9
    assert result_no_base.filter2_count == 7
    assert result_no_base.joint_count == 5


def test_get_decrypted_secret_integration_simple(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-simple-secret",
        secret_value={"api_key": "sk-test-123456"},
        provider="openai",
    )

    decrypted = store._get_decrypted_secret(secret_info.secret_id)

    assert decrypted == {"api_key": "sk-test-123456"}


def test_get_decrypted_secret_integration_compound(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-compound-secret",
        secret_value={
            "aws_access_key_id": "AKIA1234567890",
            "aws_secret_access_key": "secret-key-value",
        },
        provider="bedrock",
    )

    decrypted = store._get_decrypted_secret(secret_info.secret_id)

    assert decrypted == {
        "aws_access_key_id": "AKIA1234567890",
        "aws_secret_access_key": "secret-key-value",
    }


def test_get_decrypted_secret_integration_with_auth_config(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-auth-config-secret",
        secret_value={"api_key": "aws-secret"},
        provider="bedrock",
        auth_config={"region": "us-east-1", "profile": "default"},
    )

    decrypted = store._get_decrypted_secret(secret_info.secret_id)

    assert decrypted == {"api_key": "aws-secret"}


def test_get_decrypted_secret_integration_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store._get_decrypted_secret("nonexistent-secret-id")


def test_get_decrypted_secret_integration_multiple_secrets(store):
    secret1 = store.create_gateway_secret(
        secret_name="secret-1",
        secret_value={"api_key": "key-1"},
        provider="openai",
    )
    secret2 = store.create_gateway_secret(
        secret_name="secret-2",
        secret_value={"api_key": "key-2"},
        provider="anthropic",
    )

    decrypted1 = store._get_decrypted_secret(secret1.secret_id)
    decrypted2 = store._get_decrypted_secret(secret2.secret_id)

    assert decrypted1 == {"api_key": "key-1"}
    assert decrypted2 == {"api_key": "key-2"}
