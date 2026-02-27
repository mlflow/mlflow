import pytest

from mlflow.exceptions import MlflowException


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
    assert issue.status == "draft"
    assert issue.confidence is None
    assert issue.root_causes is None
    assert issue.source_run_id is None
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
        name="Token limit exceeded",
        description="Model is hitting token limits frequently",
        status="accepted",
        confidence="high",
        root_causes=["Input prompts are too long", "Context window exceeded"],
        source_run_id=run.info.run_id,
        created_by="user@example.com",
    )

    assert issue.issue_id.startswith("iss-")
    assert issue.experiment_id == exp_id
    assert issue.name == "Token limit exceeded"
    assert issue.description == "Model is hitting token limits frequently"
    assert issue.status == "accepted"
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
            status="draft",
        )


def test_create_issue_invalid_run(store):
    exp_id = store.create_experiment("test")

    with pytest.raises(MlflowException, match=r"FOREIGN KEY constraint failed"):
        store.create_issue(
            experiment_id=exp_id,
            source_run_id="nonexistent-run-id",
            name="Test issue",
            description="This should fail",
            status="draft",
        )


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
        name="Low accuracy",
        description="Model accuracy below threshold",
        status="draft",
        confidence="medium",
        root_causes=["Insufficient training data", "Model drift"],
        source_run_id=run.info.run_id,
        created_by="alice@example.com",
    )

    retrieved_issue = store.get_issue(created_issue.issue_id)

    # Verify all fields
    assert retrieved_issue.issue_id == created_issue.issue_id
    assert retrieved_issue.experiment_id == exp_id
    assert retrieved_issue.name == "Low accuracy"
    assert retrieved_issue.description == "Model accuracy below threshold"
    assert retrieved_issue.status == "draft"
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
    exp_id = store.create_experiment("test")

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Original name",
        description="Original description",
        status="draft",
        root_causes=["Initial root cause"],
        confidence="low",
    )

    # Update all supported fields (status, name, description, confidence)
    updated_issue = store.update_issue(
        issue_id=created_issue.issue_id,
        status="accepted",
        name="Updated name",
        description="Updated description",
        confidence="high",
    )

    # Verify updated fields
    assert updated_issue.issue_id == created_issue.issue_id
    assert updated_issue.experiment_id == exp_id
    assert updated_issue.status == "accepted"
    assert updated_issue.name == "Updated name"
    assert updated_issue.description == "Updated description"
    assert updated_issue.confidence == "high"

    # Verify other fields remain unchanged
    assert updated_issue.root_causes == ["Initial root cause"]
    assert updated_issue.source_run_id is None
    assert updated_issue.created_by == created_issue.created_by
    assert updated_issue.created_timestamp == created_issue.created_timestamp
    assert updated_issue.last_updated_timestamp > created_issue.last_updated_timestamp

    # Verify the updates are persisted by retrieving the issue again
    retrieved_issue = store.get_issue(created_issue.issue_id)
    assert retrieved_issue.status == "accepted"
    assert retrieved_issue.name == "Updated name"
    assert retrieved_issue.description == "Updated description"
    assert retrieved_issue.confidence == "high"
    assert retrieved_issue.root_causes == ["Initial root cause"]
    assert retrieved_issue.last_updated_timestamp == updated_issue.last_updated_timestamp


def test_update_issue_partial(store):
    exp_id = store.create_experiment("test")

    created_issue = store.create_issue(
        experiment_id=exp_id,
        name="Test issue",
        description="Test description",
        status="draft",
        root_causes=["Initial root cause"],
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
    assert updated_issue.root_causes == ["Initial root cause"]


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
    created_issues = []
    for i in range(5):
        issue = store.create_issue(
            experiment_id=exp_id,
            name=f"Issue {i}",
            description=f"Description {i}",
            frequency=0.9 - (i * 0.1),  # 0.9, 0.8, 0.7, 0.6, 0.5
            status="draft",
        )
        created_issues.append(issue)

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
