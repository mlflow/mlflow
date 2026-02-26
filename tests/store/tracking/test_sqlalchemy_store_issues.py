import uuid

import pytest

from mlflow.entities import TraceInfo, TraceState, trace_location
from mlflow.exceptions import MlflowException
from mlflow.utils.time import get_current_time_millis


def test_create_issue_required_fields_only(store):
    exp_id = store.create_experiment("test")

    issue = store.create_issue(
        experiment_id=exp_id,
        name="High latency",
        description="API calls are taking too long",
        frequency=0.75,
        status="draft",
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.name == "High latency"
    assert issue.description == "API calls are taking too long"
    assert issue.frequency == 0.75
    assert issue.status == "draft"
    assert issue.run_id is None
    assert issue.root_cause is None
    assert issue.confidence is None
    assert issue.rationale_examples is None
    assert issue.example_trace_ids is None
    assert issue.trace_ids is None
    assert issue.created_by is None
    assert issue.created_timestamp > 0
    assert issue.last_updated_timestamp == issue.created_timestamp


def test_create_issue_with_all_fields(store):
    exp_id = store.create_experiment("test")
    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        run_name="test_run",
        tags=[],
    )

    issue = store.create_issue(
        experiment_id=exp_id,
        run_id=run.info.run_id,
        name="Token limit exceeded",
        description="Model is hitting token limits frequently",
        frequency=0.42,
        status="accepted",
        root_cause="Input prompts are too long",
        confidence="high",
        rationale_examples='["Example 1", "Example 2"]',
        example_trace_ids='["trace-1", "trace-2"]',
        created_by="user@example.com",
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.run_id == run.info.run_id
    assert issue.name == "Token limit exceeded"
    assert issue.description == "Model is hitting token limits frequently"
    assert issue.frequency == 0.42
    assert issue.status == "accepted"
    assert issue.root_cause == "Input prompts are too long"
    assert issue.confidence == "high"
    assert issue.rationale_examples == ["Example 1", "Example 2"]
    assert issue.example_trace_ids == ["trace-1", "trace-2"]
    assert issue.created_by == "user@example.com"


def test_create_issue_with_trace_ids(store):
    exp_id = store.create_experiment("test")

    # Create two traces
    timestamp_ms = get_current_time_millis()
    trace_info_1 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
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

    trace_info_2 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
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

    issue = store.create_issue(
        experiment_id=exp_id,
        name="Authentication failure",
        description="Users are getting auth errors",
        frequency=0.15,
        status="draft",
        trace_ids=[trace_info_1.request_id, trace_info_2.request_id],
    )

    assert issue.issue_id.startswith("iss-")
    assert len(issue.trace_ids) == 2
    assert trace_info_1.request_id in issue.trace_ids
    assert trace_info_2.request_id in issue.trace_ids


def test_create_issue_invalid_experiment(store):
    with pytest.raises(MlflowException, match=r"No Experiment with id=999999 exists"):
        store.create_issue(
            experiment_id="999999",
            name="Test issue",
            description="This should fail",
            frequency=0.5,
            status="draft",
        )


def test_create_issue_invalid_run(store):
    exp_id = store.create_experiment("test")

    with pytest.raises(MlflowException, match=r"Run .* not found"):
        store.create_issue(
            experiment_id=exp_id,
            run_id="nonexistent-run-id",
            name="Test issue",
            description="This should fail",
            frequency=0.5,
            status="draft",
        )


def test_create_issue_creates_issue_reference_assessments(store):
    from mlflow.entities.assessment import IssueReference

    exp_id = store.create_experiment("test")

    # Create a trace
    timestamp_ms = get_current_time_millis()
    trace_info = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
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

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Timeout error",
        description="Request timeouts",
        frequency=0.4,
        status="draft",
        trace_ids=[trace_info.request_id],
    )

    # Verify the trace has an IssueReference assessment
    trace_info_with_assessments = store.get_trace_info(trace_info.request_id)
    assert len(trace_info_with_assessments.assessments) == 1

    assessment = trace_info_with_assessments.assessments[0]
    assert isinstance(assessment, IssueReference)
    assert assessment.issue_name == "Timeout error"
    assert assessment.issue_id == created_issue.issue_id
    assert assessment.name == created_issue.issue_id  # name field stores issue_id
    assert assessment.trace_id == trace_info.request_id
    assert assessment.source.source_type == "CODE"
    assert assessment.source.source_id == "issue_discovery"
