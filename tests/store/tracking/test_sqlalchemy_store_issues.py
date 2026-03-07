import pytest

from mlflow.entities import IssueStatus
from mlflow.exceptions import MlflowException


def test_create_issue_required_fields_only(store):
    exp_id = "0"

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
    assert issue.confidence is None
    assert issue.root_causes is None
    assert issue.source_run_id is None
    assert issue.created_by is None
    assert issue.created_timestamp > 0
    assert issue.last_updated_timestamp == issue.created_timestamp


def test_create_issue_with_all_fields(store):
    exp_id = "0"
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
        status=IssueStatus.ACCEPTED,
        confidence="high",
        root_causes=["Input prompts are too long", "Context window exceeded"],
        source_run_id=run.info.run_id,
        created_by="user@example.com",
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.name == "Token limit exceeded"
    assert issue.description == "Model is hitting token limits frequently"
    assert issue.status == IssueStatus.ACCEPTED
    assert issue.confidence == "high"
    assert issue.root_causes == ["Input prompts are too long", "Context window exceeded"]
    assert issue.source_run_id == run.info.run_id
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
    exp_id = "0"

    with pytest.raises(MlflowException, match=r"FOREIGN KEY constraint failed"):
        store.create_issue(
            experiment_id=exp_id,
            source_run_id="nonexistent-run-id",
            name="Test issue",
            description="This should fail",
            status=IssueStatus.PENDING,
        )


def test_get_issue(store):
    exp_id = "0"

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
        confidence="medium",
        root_causes=["Insufficient training data", "Model drift"],
        source_run_id=run.info.run_id,
        created_by="alice@example.com",
    )

    retrieved_issue = store.get_issue(created_issue.issue_id)

    assert retrieved_issue.issue_id == created_issue.issue_id
    assert retrieved_issue.experiment_id == exp_id
    assert retrieved_issue.name == "Low accuracy"
    assert retrieved_issue.description == "Model accuracy below threshold"
    assert retrieved_issue.status == IssueStatus.PENDING
    assert retrieved_issue.confidence == "medium"
    assert retrieved_issue.root_causes == ["Insufficient training data", "Model drift"]
    assert retrieved_issue.source_run_id == run.info.run_id
    assert retrieved_issue.created_by == "alice@example.com"
    assert retrieved_issue.created_timestamp is not None
    assert retrieved_issue.created_timestamp > 0


def test_get_issue_nonexistent(store):
    with pytest.raises(MlflowException, match=r"Issue with ID 'nonexistent-id' not found"):
        store.get_issue("nonexistent-id")


def test_update_issue(store):
    exp_id = "0"

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Original name",
        description="Original description",
        status=IssueStatus.PENDING,
        root_causes=["Initial root cause"],
        confidence="low",
    )

    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status=IssueStatus.ACCEPTED,
        name="Updated name",
        description="Updated description",
        confidence="high",
    )

    assert updated_issue.issue_id == created_issue.issue_id
    assert updated_issue.experiment_id == exp_id
    assert updated_issue.status == "accepted"
    assert updated_issue.name == "Updated name"
    assert updated_issue.description == "Updated description"
    assert updated_issue.confidence == "high"
    assert updated_issue.root_causes == ["Initial root cause"]
    assert updated_issue.source_run_id is None
    assert updated_issue.created_by == created_issue.created_by
    assert updated_issue.created_timestamp == created_issue.created_timestamp
    assert updated_issue.last_updated_timestamp > created_issue.last_updated_timestamp

    retrieved_issue = store.get_issue(created_issue.issue_id)
    assert retrieved_issue.status == "accepted"
    assert retrieved_issue.name == "Updated name"
    assert retrieved_issue.description == "Updated description"
    assert retrieved_issue.confidence == "high"
    assert retrieved_issue.root_causes == ["Initial root cause"]
    assert retrieved_issue.last_updated_timestamp == updated_issue.last_updated_timestamp


def test_update_issue_partial(store):
    exp_id = "0"

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Test issue",
        description="Test description",
        status=IssueStatus.PENDING,
        root_causes=["Initial root cause"],
    )

    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status=IssueStatus.ACCEPTED,
    )

    assert updated_issue.status == "accepted"
    assert updated_issue.name == "Test issue"
    assert updated_issue.description == "Test description"
    assert updated_issue.root_causes == ["Initial root cause"]


def test_update_issue_nonexistent(store):
    with pytest.raises(MlflowException, match=r"Issue with ID 'nonexistent-id' not found"):
        store.update_issue(issue_id="nonexistent-id", status=IssueStatus.ACCEPTED)


def test_search_issues_no_filters(store):
    exp_id = "0"

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
    exp_id = "0"

    created_issues = [
        store.create_issue(
            experiment_id=exp_id,
            name=f"Issue {i}",
            description=f"Description {i}",
            status=IssueStatus.PENDING,
        )
        for i in range(5)
    ]

    page1 = store.search_issues(max_results=2)

    assert len(page1) == 2
    assert page1[0].issue_id == created_issues[4].issue_id
    assert page1[1].issue_id == created_issues[3].issue_id
    assert page1.token is not None

    page2 = store.search_issues(max_results=2, page_token=page1.token)

    assert len(page2) == 2
    assert page2[0].issue_id == created_issues[2].issue_id
    assert page2[1].issue_id == created_issues[1].issue_id
    assert page2.token is not None

    page3 = store.search_issues(max_results=2, page_token=page2.token)

    assert len(page3) == 1
    assert page3[0].issue_id == created_issues[0].issue_id
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
    exp_id = "0"

    store.create_issue(
        experiment_id=exp_id,
        name="Pending Issue",
        description="In pending",
        status=IssueStatus.PENDING,
    )

    issue_accepted = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Issue",
        description="Accepted",
        status=IssueStatus.ACCEPTED,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Rejected Issue",
        description="Rejected",
        status=IssueStatus.REJECTED,
    )

    result = store.search_issues(filter_string="status = 'accepted'")

    assert len(result) == 1
    assert result[0].issue_id == issue_accepted.issue_id
    assert result[0].status == IssueStatus.ACCEPTED


def test_search_issues_filter_by_source_run_id(store):
    exp_id = "0"

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
        name="Exp1 Accepted",
        description="Accepted in exp1",
        status=IssueStatus.ACCEPTED,
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
    exp_id = "0"

    store.create_issue(
        experiment_id=exp_id,
        name="Test Issue",
        description="Test",
        status=IssueStatus.PENDING,
    )

    with pytest.raises(MlflowException, match=r"Invalid filter field 'invalid_field'"):
        store.search_issues(filter_string="invalid_field = 'value'")


def test_search_issues_filter_inequality(store):
    exp_id = "0"

    store.create_issue(
        experiment_id=exp_id,
        name="Draft Issue",
        description="In draft",
        status=IssueStatus.PENDING,
    )

    issue_accepted = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Issue",
        description="Accepted",
        status=IssueStatus.ACCEPTED,
    )

    result = store.search_issues(filter_string="status != 'pending'")

    assert len(result) == 1
    assert result[0].issue_id == issue_accepted.issue_id
    assert result[0].status == IssueStatus.ACCEPTED


def test_search_issues_filter_and_operator(store):
    exp_id = "0"

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

    issue_accepted_run1 = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Run1",
        description="Accepted from run1",
        status=IssueStatus.ACCEPTED,
        source_run_id=run1.info.run_id,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Accepted Run2",
        description="Accepted from run2",
        status=IssueStatus.ACCEPTED,
        source_run_id=run2.info.run_id,
    )

    result = store.search_issues(
        filter_string=f"status = 'accepted' AND source_run_id = '{run1.info.run_id}'"
    )

    assert len(result) == 1
    assert result[0].issue_id == issue_accepted_run1.issue_id
    assert result[0].status == IssueStatus.ACCEPTED
    assert result[0].source_run_id == run1.info.run_id
