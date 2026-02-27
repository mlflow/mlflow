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
        status="draft",
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.name == "High latency"
    assert issue.description == "API calls are taking too long"
    assert issue.frequency is None
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
        rationale_examples=["Example 1", "Example 2"],
        example_trace_ids=["trace-1", "trace-2"],
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


def test_get_issue(store):
    exp_id = store.create_experiment("test")

    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        run_name="test_run",
        tags=[],
    )

    created_issue = store.create_issue(
        experiment_id=exp_id,
        run_id=run.info.run_id,
        name="Low accuracy",
        description="Model accuracy below threshold",
        frequency=0.88,
        status="draft",
        root_cause="Insufficient training data",
        confidence="medium",
        rationale_examples=["Example 1", "Example 2", "Example 3"],
        example_trace_ids=["trace-a", "trace-b"],
        created_by="alice@example.com",
    )

    retrieved_issue = store.get_issue(created_issue.issue_id)

    # Verify all fields
    assert retrieved_issue.issue_id == created_issue.issue_id
    assert retrieved_issue.experiment_id == exp_id
    assert retrieved_issue.run_id == run.info.run_id
    assert retrieved_issue.name == "Low accuracy"
    assert retrieved_issue.description == "Model accuracy below threshold"
    assert retrieved_issue.frequency == 0.88
    assert retrieved_issue.status == "draft"
    assert retrieved_issue.root_cause == "Insufficient training data"
    assert retrieved_issue.confidence == "medium"
    assert retrieved_issue.rationale_examples == ["Example 1", "Example 2", "Example 3"]
    assert retrieved_issue.example_trace_ids == ["trace-a", "trace-b"]
    assert retrieved_issue.trace_ids is None
    assert retrieved_issue.created_by == "alice@example.com"
    assert retrieved_issue.created_timestamp is not None
    assert retrieved_issue.created_timestamp > 0


def test_get_issue_nonexistent(store):
    with pytest.raises(MlflowException, match=r"Issue with ID 'nonexistent-id' not found"):
        store.get_issue("nonexistent-id")


def test_get_issue_with_trace_ids(store):
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
        name="Performance issue",
        description="Slow response times",
        frequency=0.6,
        status="draft",
        trace_ids=[trace_info.request_id],
    )

    retrieved_issue = store.get_issue(created_issue.issue_id)

    assert retrieved_issue.issue_id == created_issue.issue_id
    assert len(retrieved_issue.trace_ids) == 1
    assert retrieved_issue.trace_ids[0] == trace_info.request_id


def test_update_issue(store):
    exp_id = store.create_experiment("test")

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Original name",
        description="Original description",
        frequency=0.5,
        status="draft",
        root_cause="Initial root cause",
        confidence="low",
    )

    # Update all supported fields (status, name, description)
    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status="accepted",
        name="Updated name",
        description="Updated description",
    )

    # Verify updated fields
    assert updated_issue.issue_id == created_issue.issue_id
    assert updated_issue.experiment_id == exp_id
    assert updated_issue.status == "accepted"
    assert updated_issue.name == "Updated name"
    assert updated_issue.description == "Updated description"

    # Verify other fields remain unchanged
    assert updated_issue.frequency == 0.5
    assert updated_issue.root_cause == "Initial root cause"
    assert updated_issue.confidence == "low"
    assert updated_issue.run_id is None
    assert updated_issue.created_by == created_issue.created_by
    assert updated_issue.created_timestamp == created_issue.created_timestamp
    assert updated_issue.last_updated_timestamp > created_issue.last_updated_timestamp

    # Verify the updates are persisted by retrieving the issue again
    retrieved_issue = store.get_issue(created_issue.issue_id)
    assert retrieved_issue.status == "accepted"
    assert retrieved_issue.name == "Updated name"
    assert retrieved_issue.description == "Updated description"
    assert retrieved_issue.frequency == 0.5
    assert retrieved_issue.root_cause == "Initial root cause"
    assert retrieved_issue.confidence == "low"
    assert retrieved_issue.last_updated_timestamp == updated_issue.last_updated_timestamp


def test_update_issue_partial(store):
    exp_id = store.create_experiment("test")

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Test issue",
        description="Test description",
        frequency=0.3,
        status="draft",
        root_cause="Initial root cause",
    )

    # Update only status field
    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status="accepted",
    )

    # Verify updated field changed
    assert updated_issue.status == "accepted"

    # Verify other fields unchanged
    assert updated_issue.name == "Test issue"
    assert updated_issue.description == "Test description"
    assert updated_issue.frequency == 0.3
    assert updated_issue.root_cause == "Initial root cause"


def test_update_issue_nonexistent(store):
    with pytest.raises(MlflowException, match=r"Issue with ID 'nonexistent-id' not found"):
        store.update_issue(issue_id="nonexistent-id", status="accepted")


def test_search_issues_no_filters(store):
    exp_id = store.create_experiment("test")

    # Create 3 issues with different frequencies
    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 1",
        description="High frequency",
        frequency=0.9,
        status="draft",
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 2",
        description="Medium frequency",
        frequency=0.5,
        status="draft",
    )

    issue3 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 3",
        description="Low frequency",
        frequency=0.2,
        status="draft",
    )

    # Search without filters should return all issues ordered by frequency DESC
    result = store.search_issues()

    assert len(result) == 3
    assert result[0].issue_id == issue1.issue_id
    assert result[0].frequency == 0.9
    assert result[1].issue_id == issue2.issue_id
    assert result[1].frequency == 0.5
    assert result[2].issue_id == issue3.issue_id
    assert result[2].frequency == 0.2
    assert result.token is None


def test_search_issues_by_experiment_id(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    # Create issues in different experiments
    issue1 = store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Issue",
        description="In experiment 1",
        frequency=0.8,
        status="draft",
    )

    store.create_issue(
        experiment_id=exp_id2,
        name="Exp2 Issue",
        description="In experiment 2",
        frequency=0.7,
        status="draft",
    )

    # Search by experiment_id should only return issues from that experiment
    result = store.search_issues(experiment_id=exp_id1)

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].experiment_id == exp_id1
    assert result.token is None


def test_search_issues_by_run_id(store):
    exp_id = store.create_experiment("test")

    run1 = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        run_name="run1",
        tags=[],
    )

    run2 = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        run_name="run2",
        tags=[],
    )

    # Create issues for different runs
    issue1 = store.create_issue(
        experiment_id=exp_id,
        run_id=run1.info.run_id,
        name="Run1 Issue",
        description="From run 1",
        frequency=0.6,
        status="draft",
    )

    store.create_issue(
        experiment_id=exp_id,
        run_id=run2.info.run_id,
        name="Run2 Issue",
        description="From run 2",
        frequency=0.5,
        status="draft",
    )

    # Search by run_id should only return issues from that run
    result = store.search_issues(run_id=run1.info.run_id)

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].run_id == run1.info.run_id
    assert result.token is None


def test_search_issues_by_status(store):
    exp_id = store.create_experiment("test")

    # Create issues with different statuses
    store.create_issue(
        experiment_id=exp_id,
        name="Draft Issue",
        description="In draft",
        frequency=0.8,
        status="draft",
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Issue",
        description="Accepted",
        frequency=0.7,
        status="accepted",
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Rejected Issue",
        description="Rejected",
        frequency=0.6,
        status="rejected",
    )

    # Search by status should only return issues with that status
    result = store.search_issues(status="accepted")

    assert len(result) == 1
    assert result[0].issue_id == issue2.issue_id
    assert result[0].status == "accepted"
    assert result.token is None


def test_search_issues_combined_filters(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    # Create issues in different experiments with different statuses
    issue1 = store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Draft",
        description="Draft in exp1",
        frequency=0.9,
        status="draft",
    )

    store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Accepted",
        description="Accepted in exp1",
        frequency=0.8,
        status="accepted",
    )

    store.create_issue(
        experiment_id=exp_id2,
        name="Exp2 Draft",
        description="Draft in exp2",
        frequency=0.7,
        status="draft",
    )

    # Search with combined filters
    result = store.search_issues(experiment_id=exp_id1, status="draft")

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].experiment_id == exp_id1
    assert result[0].status == "draft"
    assert result.token is None


def test_search_issues_pagination(store):
    exp_id = store.create_experiment("test")

    # Create 5 issues
    created_issues = [
        store.create_issue(
            experiment_id=exp_id,
            name=f"Issue {i}",
            description=f"Description {i}",
            frequency=0.9 - (i * 0.1),  # 0.9, 0.8, 0.7, 0.6, 0.5
            status="draft",
        )
        for i in range(5)
    ]

    # First page: get 2 results
    page1 = store.search_issues(max_results=2)

    assert len(page1) == 2
    assert page1[0].issue_id == created_issues[0].issue_id
    assert page1[1].issue_id == created_issues[1].issue_id
    assert page1.token is not None

    # Second page: get next 2 results
    page2 = store.search_issues(max_results=2, page_token=page1.token)

    assert len(page2) == 2
    assert page2[0].issue_id == created_issues[2].issue_id
    assert page2[1].issue_id == created_issues[3].issue_id
    assert page2.token is not None

    # Third page: get last result
    page3 = store.search_issues(max_results=2, page_token=page2.token)

    assert len(page3) == 1
    assert page3[0].issue_id == created_issues[4].issue_id
    assert page3.token is None


def test_search_issues_empty_results(store):
    exp_id = store.create_experiment("test")

    # Create an issue
    store.create_issue(
        experiment_id=exp_id,
        name="Test Issue",
        description="Test",
        frequency=0.5,
        status="draft",
    )

    # Search with non-matching filter
    result = store.search_issues(status="accepted")

    assert len(result) == 0
    assert result.token is None


def test_search_issues_with_null_frequency(store):
    exp_id = store.create_experiment("test")

    # Create issues with various frequencies and NULL
    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="High frequency",
        description="High frequency issue",
        frequency=0.9,
        status="draft",
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="No frequency",
        description="Issue without frequency",
        status="draft",
    )

    issue3 = store.create_issue(
        experiment_id=exp_id,
        name="Medium frequency",
        description="Medium frequency issue",
        frequency=0.5,
        status="draft",
    )

    issue4 = store.create_issue(
        experiment_id=exp_id,
        name="Another no frequency",
        description="Another issue without frequency",
        status="draft",
    )

    # Search should return issues sorted by frequency DESC (with NULLs last),
    # then by created_timestamp DESC
    result = store.search_issues()

    assert len(result) == 4
    # Issues with frequency should come first, sorted by frequency DESC
    assert result[0].issue_id == issue1.issue_id
    assert result[0].frequency == 0.9
    assert result[1].issue_id == issue3.issue_id
    assert result[1].frequency == 0.5
    # Issues with NULL frequency should come last, sorted by created_timestamp DESC
    assert result[2].issue_id == issue4.issue_id
    assert result[2].frequency is None
    assert result[3].issue_id == issue2.issue_id
    assert result[3].frequency is None
