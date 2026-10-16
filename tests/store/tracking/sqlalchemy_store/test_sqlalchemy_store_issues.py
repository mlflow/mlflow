import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

from mlflow.entities import Issue, IssueSeverity, IssueStatus, TraceState
from mlflow.entities.assessment import IssueReference
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.exceptions import MlflowException
from mlflow.utils.time import get_current_time_millis

DEFAULT_EXPERIMENT_ID = "0"


def test_create_issue_required_fields_only(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    issue = store.create_issue(
        experiment_id=exp_id,
        name="High latency",
        description="API calls are taking too long",
        status=IssueStatus.PENDING,
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.name == "High latency"
    assert issue.description == "API calls are taking too long"
    assert issue.status == IssueStatus.PENDING
    assert issue.severity is None
    assert issue.root_causes is None
    assert issue.source_run_id is None
    assert issue.categories is None
    assert issue.created_by is None
    assert issue.created_timestamp > 0
    assert issue.last_updated_timestamp == issue.created_timestamp


def test_create_issue_with_all_fields(store):
    exp_id = DEFAULT_EXPERIMENT_ID
    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        run_name="test_run",
        tags=[],
    )

    issue = store.create_issue(
        experiment_id=exp_id,
        name="Token limit exceeded",
        description="Model is hitting token limits frequently",
        status=IssueStatus.RESOLVED,
        severity=IssueSeverity.HIGH,
        root_causes=["Input prompts are too long", "Context window exceeded"],
        source_run_id=run.info.run_id,
        categories=["hallucination", "context_limit"],
        created_by="user@example.com",
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.name == "Token limit exceeded"
    assert issue.description == "Model is hitting token limits frequently"
    assert issue.status == IssueStatus.RESOLVED
    assert issue.severity == IssueSeverity.HIGH
    assert issue.root_causes == [
        "Input prompts are too long",
        "Context window exceeded",
    ]
    assert issue.source_run_id == run.info.run_id
    assert issue.categories == ["hallucination", "context_limit"]
    assert issue.created_by == "user@example.com"


def test_create_issue_invalid_experiment(store):
    with pytest.raises(MlflowException, match=r"No Experiment with id=999999 exists"):
        store.create_issue(
            experiment_id="999999",
            name="Test issue",
            description="This should fail",
            status=IssueStatus.PENDING,
        )


def test_create_issue_invalid_run(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    with pytest.raises(MlflowException, match="Run with id=nonexistent-run-id not found"):
        store.create_issue(
            experiment_id=exp_id,
            source_run_id="nonexistent-run-id",
            name="Test issue",
            description="This should fail",
            status=IssueStatus.PENDING,
        )


def test_get_issue(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        run_name="test_run",
        tags=[],
    )

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Low accuracy",
        description="Model accuracy below threshold",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.MEDIUM,
        root_causes=["Insufficient training data", "Model drift"],
        source_run_id=run.info.run_id,
        categories=["accuracy", "model_drift"],
        created_by="alice@example.com",
    )

    retrieved_issue = store.get_issue(created_issue.issue_id)

    assert retrieved_issue.issue_id == created_issue.issue_id
    assert retrieved_issue.experiment_id == exp_id
    assert retrieved_issue.name == "Low accuracy"
    assert retrieved_issue.description == "Model accuracy below threshold"
    assert retrieved_issue.status == IssueStatus.PENDING
    assert retrieved_issue.severity == IssueSeverity.MEDIUM
    assert retrieved_issue.root_causes == ["Insufficient training data", "Model drift"]
    assert retrieved_issue.source_run_id == run.info.run_id
    assert retrieved_issue.categories == ["accuracy", "model_drift"]
    assert retrieved_issue.created_by == "alice@example.com"
    assert retrieved_issue.created_timestamp is not None
    assert retrieved_issue.created_timestamp > 0


def test_get_issue_nonexistent(store):
    with pytest.raises(MlflowException, match=r"Issue with ID 'nonexistent-id' not found"):
        store.get_issue("nonexistent-id")


def test_update_issue(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Original name",
        description="Original description",
        status=IssueStatus.PENDING,
        root_causes=["Initial root cause"],
        severity=IssueSeverity.LOW,
    )

    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status=IssueStatus.RESOLVED,
        name="Updated name",
        description="Updated description",
        severity=IssueSeverity.HIGH,
    )

    assert updated_issue.issue_id == created_issue.issue_id
    assert updated_issue.experiment_id == exp_id
    assert updated_issue.status == IssueStatus.RESOLVED
    assert updated_issue.name == "Updated name"
    assert updated_issue.description == "Updated description"
    assert updated_issue.severity == IssueSeverity.HIGH
    assert updated_issue.root_causes == ["Initial root cause"]
    assert updated_issue.source_run_id is None
    assert updated_issue.created_by == created_issue.created_by
    assert updated_issue.created_timestamp == created_issue.created_timestamp
    assert updated_issue.last_updated_timestamp > created_issue.last_updated_timestamp

    retrieved_issue = store.get_issue(created_issue.issue_id)
    assert retrieved_issue.status == IssueStatus.RESOLVED
    assert retrieved_issue.name == "Updated name"
    assert retrieved_issue.description == "Updated description"
    assert retrieved_issue.severity == IssueSeverity.HIGH
    assert retrieved_issue.root_causes == ["Initial root cause"]
    assert retrieved_issue.last_updated_timestamp == updated_issue.last_updated_timestamp


def test_update_issue_partial(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Test issue",
        description="Test description",
        status=IssueStatus.PENDING,
        root_causes=["Initial root cause"],
    )

    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status=IssueStatus.RESOLVED,
    )

    assert updated_issue.status == "resolved"
    assert updated_issue.name == "Test issue"
    assert updated_issue.description == "Test description"
    assert updated_issue.root_causes == ["Initial root cause"]


def test_update_issue_nonexistent(store):
    with pytest.raises(MlflowException, match=r"Issue with ID 'nonexistent-id' not found"):
        store.update_issue(issue_id="nonexistent-id", status=IssueStatus.RESOLVED)


def test_search_issues_no_filters(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 1",
        description="First issue",
        status=IssueStatus.PENDING,
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 2",
        description="Second issue",
        status=IssueStatus.PENDING,
    )

    issue3 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 3",
        description="Third issue",
        status=IssueStatus.PENDING,
    )

    result = store.search_issues()

    assert len(result) == 3
    assert result[0].issue_id == issue3.issue_id
    assert result[1].issue_id == issue2.issue_id
    assert result[2].issue_id == issue1.issue_id
    assert result.token is None


def test_search_issues_sorted_by_severity(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    issue_low = store.create_issue(
        experiment_id=exp_id,
        name="Low severity issue",
        description="Low severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.LOW,
    )

    issue_high = store.create_issue(
        experiment_id=exp_id,
        name="High severity issue",
        description="High severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.HIGH,
    )

    issue_medium = store.create_issue(
        experiment_id=exp_id,
        name="Medium severity issue",
        description="Medium severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.MEDIUM,
    )

    issue_none = store.create_issue(
        experiment_id=exp_id,
        name="No severity issue",
        description="No severity",
        status=IssueStatus.PENDING,
    )

    issue_not_an_issue = store.create_issue(
        experiment_id=exp_id,
        name="Not an issue",
        description="Not an issue severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.NOT_AN_ISSUE,
    )

    result = store.search_issues()

    assert len(result) == 5
    assert [issue.issue_id for issue in result] == [
        issue_high.issue_id,
        issue_medium.issue_id,
        issue_low.issue_id,
        issue_not_an_issue.issue_id,
        issue_none.issue_id,
    ]


def test_search_issues_by_experiment_id(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    issue1 = store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Issue",
        description="In experiment 1",
        status=IssueStatus.PENDING,
    )

    store.create_issue(
        experiment_id=exp_id2,
        name="Exp2 Issue",
        description="In experiment 2",
        status=IssueStatus.PENDING,
    )

    result = store.search_issues(experiment_id=exp_id1)

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].experiment_id == exp_id1
    assert result.token is None


def test_search_issues_pagination(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    for i in range(5):
        store.create_issue(
            experiment_id=exp_id,
            name=f"Issue {i}",
            description=f"Description {i}",
            status=IssueStatus.PENDING,
        )

    page1 = store.search_issues(max_results=2)

    assert len(page1) == 2
    assert page1.token is not None

    page2 = store.search_issues(max_results=2, page_token=page1.token)

    assert len(page2) == 2
    assert page2.token is not None

    page3 = store.search_issues(max_results=2, page_token=page2.token)

    assert len(page3) == 1
    assert page3.token is None


def test_search_issues_empty_results(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    store.create_issue(
        experiment_id=exp_id1,
        name="Test Issue",
        description="Test",
        status=IssueStatus.PENDING,
    )

    result = store.search_issues(experiment_id=exp_id2)

    assert len(result) == 0
    assert result.token is None


def test_search_issues_filter_by_status(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    store.create_issue(
        experiment_id=exp_id,
        name="Pending Issue",
        description="In pending",
        status=IssueStatus.PENDING,
    )

    issue_resolved = store.create_issue(
        experiment_id=exp_id,
        name="Resolved Issue",
        description="Resolved",
        status=IssueStatus.RESOLVED,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Rejected Issue",
        description="Rejected",
        status=IssueStatus.REJECTED,
    )

    result = store.search_issues(filter_string="status = 'resolved'")

    assert len(result) == 1
    assert result[0].issue_id == issue_resolved.issue_id
    assert result[0].status == IssueStatus.RESOLVED


def test_search_issues_filter_by_source_run_id(store):
    exp_id = DEFAULT_EXPERIMENT_ID

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

    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="Run1 Issue",
        description="From run 1",
        status=IssueStatus.PENDING,
        source_run_id=run1.info.run_id,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Run2 Issue",
        description="From run 2",
        status=IssueStatus.PENDING,
        source_run_id=run2.info.run_id,
    )

    result = store.search_issues(filter_string=f"source_run_id = '{run1.info.run_id}'")

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].source_run_id == run1.info.run_id


def test_search_issues_filter_combined_with_experiment_id(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    issue1 = store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Draft",
        description="Draft in exp1",
        status=IssueStatus.PENDING,
    )

    store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Resolved",
        description="Resolved in exp1",
        status=IssueStatus.RESOLVED,
    )

    store.create_issue(
        experiment_id=exp_id2,
        name="Exp2 Draft",
        description="Draft in exp2",
        status=IssueStatus.PENDING,
    )

    result = store.search_issues(experiment_id=exp_id1, filter_string="status = 'pending'")

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].experiment_id == exp_id1
    assert result[0].status == IssueStatus.PENDING


def test_search_issues_filter_invalid_field(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    store.create_issue(
        experiment_id=exp_id,
        name="Test Issue",
        description="Test",
        status=IssueStatus.PENDING,
    )

    with pytest.raises(MlflowException, match=r"Invalid filter field 'invalid_field'"):
        store.search_issues(filter_string="invalid_field = 'value'")


def test_search_issues_filter_inequality(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    store.create_issue(
        experiment_id=exp_id,
        name="Draft Issue",
        description="In draft",
        status=IssueStatus.PENDING,
    )

    issue_resolved = store.create_issue(
        experiment_id=exp_id,
        name="Resolved Issue",
        description="Resolved",
        status=IssueStatus.RESOLVED,
    )

    result = store.search_issues(filter_string="status != 'pending'")

    assert len(result) == 1
    assert result[0].issue_id == issue_resolved.issue_id
    assert result[0].status == IssueStatus.RESOLVED


def test_search_issues_filter_and_operator(store):
    exp_id = DEFAULT_EXPERIMENT_ID

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

    store.create_issue(
        experiment_id=exp_id,
        name="Pending Run1",
        description="Pending from run1",
        status=IssueStatus.PENDING,
        source_run_id=run1.info.run_id,
    )

    issue_resolved_run1 = store.create_issue(
        experiment_id=exp_id,
        name="Resolved Run1",
        description="Resolved from run1",
        status=IssueStatus.RESOLVED,
        source_run_id=run1.info.run_id,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Resolved Run2",
        description="Resolved from run2",
        status=IssueStatus.RESOLVED,
        source_run_id=run2.info.run_id,
    )

    result = store.search_issues(
        filter_string=f"status = 'resolved' AND source_run_id = '{run1.info.run_id}'"
    )

    assert len(result) == 1
    assert result[0].issue_id == issue_resolved_run1.issue_id
    assert result[0].status == IssueStatus.RESOLVED
    assert result[0].source_run_id == run1.info.run_id


def test_search_issues_invalid_max_results(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    store.create_issue(
        experiment_id=exp_id,
        name="Test Issue",
        description="Test",
        status=IssueStatus.RESOLVED,
    )

    with pytest.raises(MlflowException, match=r"Invalid value 0 for parameter 'max_results'"):
        store.search_issues(max_results=0)

    with pytest.raises(MlflowException, match=r"Invalid value -1 for parameter 'max_results'"):
        store.search_issues(max_results=-1)


def test_search_traces_by_issue_id(store):
    exp_id = store.create_experiment("test")

    timestamp_ms = get_current_time_millis()
    trace1 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    trace2 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms + 100,
            execution_duration=200,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    trace3 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms + 200,
            execution_duration=300,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="High latency",
        description="API calls are slow",
        status=IssueStatus.PENDING,
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="Authentication failure",
        description="Auth errors",
        status=IssueStatus.PENDING,
    )

    store.create_assessment(
        IssueReference(
            issue_id=issue1.issue_id,
            issue_name=issue1.name,
            trace_id=trace1.request_id,
        )
    )
    store.create_assessment(
        IssueReference(
            issue_id=issue1.issue_id,
            issue_name=issue1.name,
            trace_id=trace2.request_id,
        )
    )
    store.create_assessment(
        IssueReference(
            issue_id=issue2.issue_id,
            issue_name=issue2.name,
            trace_id=trace3.request_id,
        )
    )

    traces, _ = store.search_traces(
        experiment_ids=[exp_id],
        filter_string=f"issue.id = '{issue1.issue_id}'",
    )

    assert len(traces) == 2
    trace_ids = {t.request_id for t in traces}
    assert trace_ids == {trace1.request_id, trace2.request_id}

    traces, _ = store.search_traces(
        experiment_ids=[exp_id],
        filter_string=f"issue.id = '{issue2.issue_id}'",
    )

    assert len(traces) == 1
    assert traces[0].request_id == trace3.request_id

    traces, _ = store.search_traces(
        experiment_ids=[exp_id],
        filter_string="issue.id = 'non-existent-issue'",
    )
    assert len(traces) == 0


def test_search_issues_without_trace_count(store):
    exp_id = store.create_experiment("test")

    issue = store.create_issue(
        experiment_id=exp_id,
        name="Test Issue",
        description="Test",
        status=IssueStatus.PENDING,
    )

    timestamp_ms = get_current_time_millis()
    trace = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    store.create_assessment(
        IssueReference(
            issue_id=issue.issue_id,
            issue_name=issue.name,
            trace_id=trace.request_id,
        )
    )

    result = store.search_issues(experiment_id=exp_id)

    assert len(result) == 1
    assert result[0].issue_id == issue.issue_id
    assert result[0].trace_count is None


def test_search_issues_with_trace_count(store):
    exp_id = store.create_experiment("test")

    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="Issue with 2 traces",
        description="Has 2 traces",
        status=IssueStatus.PENDING,
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="Issue with 1 trace",
        description="Has 1 trace",
        status=IssueStatus.PENDING,
    )

    issue3 = store.create_issue(
        experiment_id=exp_id,
        name="Issue with 0 traces",
        description="Has no traces",
        status=IssueStatus.PENDING,
    )

    timestamp_ms = get_current_time_millis()
    trace1 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    trace2 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms + 100,
            execution_duration=200,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    trace3 = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms + 200,
            execution_duration=300,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    store.create_assessment(
        IssueReference(
            issue_id=issue1.issue_id,
            issue_name=issue1.name,
            trace_id=trace1.request_id,
        )
    )
    store.create_assessment(
        IssueReference(
            issue_id=issue1.issue_id,
            issue_name=issue1.name,
            trace_id=trace2.request_id,
        )
    )
    store.create_assessment(
        IssueReference(
            issue_id=issue2.issue_id,
            issue_name=issue2.name,
            trace_id=trace3.request_id,
        )
    )

    result = store.search_issues(experiment_id=exp_id, include_trace_count=True)

    assert len(result) == 3

    issues_by_id = {issue.issue_id: issue for issue in result}

    assert issues_by_id[issue1.issue_id].trace_count == 2
    assert issues_by_id[issue2.issue_id].trace_count == 1
    assert issues_by_id[issue3.issue_id].trace_count == 0


def test_search_issues_with_trace_count_pagination(store):
    exp_id = store.create_experiment("test")

    for i in range(3):
        issue = store.create_issue(
            experiment_id=exp_id,
            name=f"Issue {i}",
            description=f"Description {i}",
            status=IssueStatus.PENDING,
        )

        timestamp_ms = get_current_time_millis()
        for j in range(i + 1):
            trace = store.start_trace(
                TraceInfo(
                    trace_id=f"tr-{uuid.uuid4()}",
                    trace_location=TraceLocation.from_experiment_id(exp_id),
                    request_time=timestamp_ms + j,
                    execution_duration=100,
                    state=TraceState.OK,
                    tags={},
                    trace_metadata={},
                    client_request_id=f"tr-{uuid.uuid4()}",
                    request_preview=None,
                    response_preview=None,
                ),
            )

            store.create_assessment(
                IssueReference(
                    issue_id=issue.issue_id,
                    issue_name=issue.name,
                    trace_id=trace.request_id,
                )
            )

    page1 = store.search_issues(experiment_id=exp_id, max_results=2, include_trace_count=True)

    assert len(page1) == 2
    assert page1[0].trace_count is not None
    assert page1[1].trace_count is not None
    assert page1.token is not None

    page2 = store.search_issues(
        experiment_id=exp_id, max_results=2, page_token=page1.token, include_trace_count=True
    )

    assert len(page2) == 1
    assert page2[0].trace_count is not None
    assert page2.token is None


def test_search_issues_trace_count_with_filters(store):
    exp_id = DEFAULT_EXPERIMENT_ID

    issue_pending = store.create_issue(
        experiment_id=exp_id,
        name="Pending Issue",
        description="Pending",
        status=IssueStatus.PENDING,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Resolved Issue",
        description="Resolved",
        status=IssueStatus.RESOLVED,
    )

    timestamp_ms = get_current_time_millis()
    trace = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    store.create_assessment(
        IssueReference(
            issue_id=issue_pending.issue_id,
            issue_name=issue_pending.name,
            trace_id=trace.request_id,
        )
    )

    result = store.search_issues(
        experiment_id=exp_id, filter_string="status = 'pending'", include_trace_count=True
    )

    assert len(result) == 1
    assert result[0].issue_id == issue_pending.issue_id
    assert result[0].trace_count == 1


def test_search_issues_sorted_by_trace_count_within_severity(store):
    exp_id = store.create_experiment("test")

    issue_high_3_traces = store.create_issue(
        experiment_id=exp_id,
        name="High severity with 3 traces",
        description="High severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.HIGH,
    )

    issue_high_1_trace = store.create_issue(
        experiment_id=exp_id,
        name="High severity with 1 trace",
        description="High severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.HIGH,
    )

    issue_medium_2_traces = store.create_issue(
        experiment_id=exp_id,
        name="Medium severity with 2 traces",
        description="Medium severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.MEDIUM,
    )

    issue_medium_0_traces = store.create_issue(
        experiment_id=exp_id,
        name="Medium severity with 0 traces",
        description="Medium severity",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.MEDIUM,
    )

    timestamp_ms = get_current_time_millis()

    for i in range(3):
        trace = store.start_trace(
            TraceInfo(
                trace_id=f"tr-{uuid.uuid4()}",
                trace_location=TraceLocation.from_experiment_id(exp_id),
                request_time=timestamp_ms + i * 100,
                execution_duration=100,
                state=TraceState.OK,
                tags={},
                trace_metadata={},
                client_request_id=f"tr-{uuid.uuid4()}",
                request_preview=None,
                response_preview=None,
            ),
        )
        store.create_assessment(
            IssueReference(
                issue_id=issue_high_3_traces.issue_id,
                issue_name=issue_high_3_traces.name,
                trace_id=trace.request_id,
            )
        )

    trace = store.start_trace(
        TraceInfo(
            trace_id=f"tr-{uuid.uuid4()}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms + 300,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )
    store.create_assessment(
        IssueReference(
            issue_id=issue_high_1_trace.issue_id,
            issue_name=issue_high_1_trace.name,
            trace_id=trace.request_id,
        )
    )

    for i in range(2):
        trace = store.start_trace(
            TraceInfo(
                trace_id=f"tr-{uuid.uuid4()}",
                trace_location=TraceLocation.from_experiment_id(exp_id),
                request_time=timestamp_ms + 400 + i * 100,
                execution_duration=100,
                state=TraceState.OK,
                tags={},
                trace_metadata={},
                client_request_id=f"tr-{uuid.uuid4()}",
                request_preview=None,
                response_preview=None,
            ),
        )
        store.create_assessment(
            IssueReference(
                issue_id=issue_medium_2_traces.issue_id,
                issue_name=issue_medium_2_traces.name,
                trace_id=trace.request_id,
            )
        )

    result = store.search_issues(experiment_id=exp_id, include_trace_count=True)

    assert len(result) == 4
    assert result[0].issue_id == issue_high_3_traces.issue_id
    assert result[0].trace_count == 3
    assert result[1].issue_id == issue_high_1_trace.issue_id
    assert result[1].trace_count == 1
    assert result[2].issue_id == issue_medium_2_traces.issue_id
    assert result[2].trace_count == 2
    assert result[3].issue_id == issue_medium_0_traces.issue_id
    assert result[3].trace_count == 0


def test_issue_extended_contract_and_partial_updates(store):
    issue = store.create_issue(
        "0",
        "Latency",
        "Details",
        issue_id="iss-caller-chosen",
        last_seen_timestamp=0,
        created_by="author",
        categories=["latency", "cost"],
        metadata={"keep": "1", "change": "old", "remove": "old"},
        root_causes=["Legacy cause"],
    )
    assert issue.last_seen_timestamp == 0
    assert issue.last_updated_by == "author"
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_issue("0", "Retry", "", issue_id=issue.issue_id)
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    updated = store.update_issue(
        issue.issue_id,
        description="",
        categories=[],
        last_seen_timestamp=10,
        last_updated_by="editor",
        metadata={"change": "new"},
        delete_metadata_keys=["remove"],
    )
    assert updated.metadata == {"keep": "1", "change": "new"}
    assert updated.description == ""
    assert updated.categories is None
    assert updated.root_causes == ["Legacy cause"]
    assert updated.created_by == "author"
    assert updated.last_updated_by == "editor"
    assert updated.created_timestamp == issue.created_timestamp
    assert store.update_issue(issue.issue_id, last_seen_timestamp=5).last_seen_timestamp == 10
    assert store.update_issue(issue.issue_id, name="New name").last_seen_timestamp == 10
    assert store.search_issues(categories=["cost"]) == []


@pytest.mark.parametrize("actor", ["", "a" * 255], ids=["empty", "max-length"])
def test_issue_create_actor(store, actor):
    issue = store.create_issue("0", "Issue", "", created_by=actor)
    for result in (issue, store.get_issue(issue.issue_id)):
        assert result.created_by == actor
        assert result.last_updated_by == actor


@pytest.mark.parametrize("actor", ["", "a" * 255], ids=["empty", "max-length"])
def test_issue_update_actor(store, actor):
    issue = store.create_issue("0", "Issue", "", created_by="author")
    updated = store.update_issue(issue.issue_id, last_updated_by=actor)
    for result in (updated, store.get_issue(issue.issue_id)):
        assert result.created_by == "author"
        assert result.last_updated_by == actor


@pytest.mark.parametrize("actor", [123, "a" * 256], ids=["non-string", "too-long"])
def test_issue_update_actor_validation_is_atomic(store, actor):
    issue = store.create_issue("0", "Original", "", created_by="author")
    with pytest.raises(MlflowException, match="Invalid last_updated_by") as exc:
        store.update_issue(issue.issue_id, name="Changed", last_updated_by=actor)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    unchanged = store.get_issue(issue.issue_id)
    assert unchanged.name == "Original"
    assert unchanged.last_updated_by == "author"


@pytest.mark.parametrize("categories", [None, [], ["cost"]])
def test_issue_empty_categories_match_create_and_clear(store, categories):
    issue = store.create_issue("0", "Issue", "", categories=categories)
    assert issue.categories == (categories or None)
    unchanged = store.update_issue(issue.issue_id, name="Renamed", categories=None)
    assert unchanged.categories == issue.categories
    matching_ids = [result.issue_id for result in store.search_issues(categories=["cost"])]
    assert matching_ids == ([issue.issue_id] if categories else [])

    cleared = store.update_issue(issue.issue_id, categories=[])
    for result in (cleared, store.get_issue(issue.issue_id), *store.search_issues()):
        assert result.categories is None
        assert Issue.from_proto(result.to_proto()).categories is None
    filtered = store.search_issues(categories=["cost"], include_total_count=True)
    assert filtered == []
    assert filtered.total_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"issue_id": "invalid"},
        {"last_seen_timestamp": -1},
        {"last_seen_timestamp": True},
        {"metadata": {"key": 123}},
        {"categories": "cost"},
        {"status": "invalid"},
        {"severity": "critical"},
        {"created_by": 123},
        {"created_by": "a" * 256},
    ],
)
def test_issue_create_validation(store, kwargs):
    with pytest.raises(MlflowException, match=".+") as exc:
        store.create_issue(**{"experiment_id": "0", "name": "Issue", "description": "", **kwargs})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_issue_patch_validation_is_atomic(store):
    issue = store.create_issue("0", "Original", "", metadata={"owner": "team"})
    with pytest.raises(MlflowException, match="same key"):
        store.update_issue(
            issue.issue_id,
            name="Changed",
            metadata={"owner": "other"},
            delete_metadata_keys=["owner"],
        )
    assert store.get_issue(issue.issue_id).name == "Original"
    with pytest.raises(MlflowException, match="at most 100"):
        store.update_issue(
            issue.issue_id, name="Changed", metadata={str(i): "v" for i in range(100)}
        )
    assert store.get_issue(issue.issue_id).metadata == {"owner": "team"}
    assert store.get_issue(issue.issue_id).name == "Original"


@pytest.mark.parametrize("value", ["x" * 1000, "\x00" * 1000], ids=["ascii", "json-escaped"])
def test_issue_large_metadata_round_trip(store, value):
    metadata = {str(index): value for index in range(70)}
    issue = store.create_issue("0", "Large metadata", "", metadata=metadata)
    assert store.get_issue(issue.issue_id).metadata == metadata
    patch = {str(index): value for index in range(70, 100)}
    store.update_issue(issue.issue_id, metadata=patch)
    assert store.get_issue(issue.issue_id).metadata == metadata | patch


def test_issue_name_search_treats_brackets_literally(store):
    issue = store.create_issue("0", "Error [500]", "")
    store.create_issue("0", "Error 5", "")
    results = store.search_issues(name="error [500]", include_total_count=True)
    assert [result.issue_id for result in results] == [issue.issue_id]
    assert results.total_count == 1


@pytest.mark.parametrize("category", ["Cost", "CustomerSupport", "Café"])
@pytest.mark.parametrize("include_trace_count", [False, True])
def test_issue_category_search_ignores_case_and_surrounding_whitespace(
    store, category, include_trace_count
):
    categories = [category, category.upper(), f" {category.lower()} "]
    issue = store.create_issue("0", "Issue", "", categories=categories)
    store.create_issue("0", "Different category", "", categories=[f"{category}Extra"])
    assert store.get_issue(issue.issue_id).categories == categories

    options = {"include_total_count": True, "include_trace_count": include_trace_count}
    results = store.search_issues(categories=[f" {category.upper()} "], **options)
    assert [result.issue_id for result in results] == [issue.issue_id]
    assert results.total_count == 1

    # Reuse the index entry when only the spelling changes, including repeated updates.
    for replacement in ([category.upper()], [f" {category.lower()} ", category]):
        assert store.update_issue(issue.issue_id, categories=replacement).categories == replacement
        results = store.search_issues(categories=[category.lower()], **options)
        assert [result.issue_id for result in results] == [issue.issue_id]
        assert results.total_count == 1

    store.update_issue(issue.issue_id, categories=["replacement"])
    assert store.search_issues(categories=[category], **options).total_count == 0
    store.update_issue(issue.issue_id, categories=[])
    assert store.search_issues(categories=["REPLACEMENT"], **options).total_count == 0


def test_issue_category_paging_accepts_equivalent_filters(store):
    for index in range(3):
        store.create_issue("0", "Issue", "", issue_id=f"iss-{index}", categories=["Cost", "COST"])
    options = {"max_results": 1, "include_total_count": True}
    first = store.search_issues(categories=["cost"], **options)
    second = store.search_issues(categories=[" COST "], page_token=first.token, **options)
    third = store.search_issues(categories=["cost", "Cost"], page_token=second.token, **options)
    assert [issue.issue_id for issue in first + second + third] == ["iss-2", "iss-1", "iss-0"]
    assert first.total_count == second.total_count == third.total_count == 3
    assert third.token is None


def test_issue_source_run_ownership(store):
    experiment_id = store.create_experiment("other")
    run = store.create_run(experiment_id, "user", 1, [], "run")
    with pytest.raises(MlflowException, match="belong"):
        store.create_issue("0", "Issue", "", source_run_id=run.info.run_id)
    issue = store.create_issue(experiment_id, "Issue", "", source_run_id=run.info.run_id)
    assert issue.source_run_id == run.info.run_id
    assert store.update_issue(issue.issue_id, name="Updated").source_run_id == run.info.run_id


def test_issue_search_facets_pagination_and_projection(store):
    for index in range(5):
        store.create_issue(
            "0",
            f"Latency 100%_{index}",
            "Large description",
            issue_id=f"iss-{index}",
            severity=IssueSeverity.HIGH,
            categories=["latency", "cost"],
            metadata={"evidence": "large"},
            root_causes=["Legacy cause"],
            last_seen_timestamp=100,
        )
    store.create_issue("0", "Latency 100xxx", "", severity=IssueSeverity.LOW)
    options = {
        "experiment_id": "0",
        "name": "LATENCY 100%_",
        "statuses": ["pending", "resolved"],
        "severities": ["high"],
        "categories": ["cost", "other"],
        "order_by": ["last_seen_timestamp DESC"],
        "include_total_count": True,
        "include_trace_count": True,
        "view": "basic",
        "max_results": 2,
    }
    first = store.search_issues(**options)
    assert [issue.issue_id for issue in first] == ["iss-4", "iss-3"]
    assert first.total_count == 5
    assert first[0].trace_count == 0
    assert first[0].description == ""
    assert first[0].root_causes is None
    assert first[0].metadata is None
    second = store.search_issues(**options, page_token=first.token)
    third = store.search_issues(**options, page_token=second.token)
    assert [issue.issue_id for issue in second + third] == ["iss-2", "iss-1", "iss-0"]
    assert third.token is None
    full = store.get_issue("iss-0")
    assert full.description == "Large description"
    assert full.metadata == {"evidence": "large"}
    for changed in [
        {"name": "different"},
        {"view": "full"},
        {"order_by": ["name ASC"]},
        {"experiment_id": "123"},
    ]:
        with pytest.raises(MlflowException, match="page token"):
            store.search_issues(**(options | changed), page_token=first.token)
    store.update_issue("iss-4", categories=["other"], status=IssueStatus.RESOLVED)
    filtered = store.search_issues(
        statuses=["resolved"], categories=["other"], include_total_count=True
    )
    assert [issue.issue_id for issue in filtered] == ["iss-4"]
    assert filtered.total_count == 1


@pytest.mark.parametrize("include_trace_count", [False, True])
def test_issue_default_and_explicit_sort_use_same_tiebreaker(store, include_trace_count):
    with mock.patch(
        "mlflow.store.tracking.sqlalchemy_store.get_current_time_millis", return_value=100
    ) as clock:
        for issue_id in ["iss-b", "iss-c", "iss-a"]:
            store.create_issue(
                "0", "Same title", "", issue_id=issue_id, severity=IssueSeverity.HIGH
            )
        assert clock.call_count == 3
    for order_by in [None, ["severity DESC", "created_timestamp DESC"]]:
        results = []
        token = None
        while True:
            page = store.search_issues(
                order_by=order_by,
                include_trace_count=include_trace_count,
                max_results=1,
                page_token=token,
            )
            results.extend(issue.issue_id for issue in page)
            if not (token := page.token):
                break
        assert results == ["iss-c", "iss-b", "iss-a"]


@pytest.mark.parametrize(
    ("order_by", "expected"),
    [
        (["severity DESC"], ["iss-high", "iss-low", "iss-null"]),
        (["status ASC"], ["iss-low", "iss-null", "iss-high"]),
        (["name DESC"], ["iss-null", "iss-low", "iss-high"]),
        (["last_seen_timestamp ASC"], ["iss-null", "iss-low", "iss-high"]),
    ],
)
def test_issue_explicit_ordering(store, order_by, expected):
    store.create_issue(
        "0",
        "A",
        "",
        issue_id="iss-high",
        severity=IssueSeverity.HIGH,
        status=IssueStatus.RESOLVED,
        last_seen_timestamp=3,
    )
    store.create_issue(
        "0", "B", "", issue_id="iss-low", severity=IssueSeverity.LOW, last_seen_timestamp=2
    )
    store.create_issue(
        "0", "C", "", issue_id="iss-null", status=IssueStatus.REJECTED, last_seen_timestamp=1
    )
    assert [issue.issue_id for issue in store.search_issues(order_by=order_by)] == expected


@pytest.mark.parametrize(
    "options",
    [
        {"name": 1},
        {"statuses": "pending"},
        {"severities": ["critical"]},
        {"order_by": "name DESC"},
        {"order_by": ["unknown DESC"]},
        {"order_by": ["name DESC extra"]},
        {"view": "invalid"},
        {"page_token": "not base64"},
        {"page_token": "bnVsbA=="},
        {"page_token": "W10="},
        {"page_token": "eyJvZmZzZXQiOi0xfQ=="},
    ],
)
def test_issue_search_validation(store, options):
    with pytest.raises(MlflowException, match=".+") as exc:
        store.search_issues(**options)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_issue_concurrent_metadata_patches_preserve_unrelated_keys(store):
    issue = store.create_issue("0", "Issue", "", last_seen_timestamp=0)

    def patch(index):
        store.update_issue(
            issue.issue_id, metadata={str(index): "value"}, last_seen_timestamp=index
        )

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="issue-patch") as executor:
        list(executor.map(patch, range(12)))
    updated = store.get_issue(issue.issue_id)
    assert updated.metadata == {str(index): "value" for index in range(12)}
    assert updated.last_seen_timestamp == 11
