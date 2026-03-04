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

    # Create 3 issues
    issue1 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 1",
        description="First issue",
        status="draft",
    )

    issue2 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 2",
        description="Second issue",
        status="draft",
    )

    issue3 = store.create_issue(
        experiment_id=exp_id,
        name="Issue 3",
        description="Third issue",
        status="draft",
    )

    # Search without filters should return all issues ordered by created_timestamp DESC
    result = store.search_issues()

    assert len(result) == 3
    # Most recent first
    assert result[0].issue_id == issue3.issue_id
    assert result[1].issue_id == issue2.issue_id
    assert result[2].issue_id == issue1.issue_id
    assert result.token is None


def test_search_issues_by_experiment_id(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    # Create issues in different experiments
    issue1 = store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Issue",
        description="In experiment 1",
        status="draft",
    )

    store.create_issue(
        experiment_id=exp_id2,
        name="Exp2 Issue",
        description="In experiment 2",
        status="draft",
    )

    # Search by experiment_id should only return issues from that experiment
    result = store.search_issues(experiment_id=exp_id1)

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].experiment_id == exp_id1
    assert result.token is None


def test_search_issues_pagination(store):
    exp_id = store.create_experiment("test")

    # Create 5 issues
    created_issues = [
        store.create_issue(
            experiment_id=exp_id,
            name=f"Issue {i}",
            description=f"Description {i}",
            status="draft",
        )
        for i in range(5)
    ]

    # First page: get 2 results (most recent first)
    page1 = store.search_issues(max_results=2)

    assert len(page1) == 2
    assert page1[0].issue_id == created_issues[4].issue_id  # Most recent
    assert page1[1].issue_id == created_issues[3].issue_id
    assert page1.token is not None

    # Second page: get next 2 results
    page2 = store.search_issues(max_results=2, page_token=page1.token)

    assert len(page2) == 2
    assert page2[0].issue_id == created_issues[2].issue_id
    assert page2[1].issue_id == created_issues[1].issue_id
    assert page2.token is not None

    # Third page: get last result
    page3 = store.search_issues(max_results=2, page_token=page2.token)

    assert len(page3) == 1
    assert page3[0].issue_id == created_issues[0].issue_id  # Oldest
    assert page3.token is None


def test_search_issues_empty_results(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    # Create an issue in exp_id1
    store.create_issue(
        experiment_id=exp_id1,
        name="Test Issue",
        description="Test",
        status="draft",
    )

    # Search in different experiment should return no results
    result = store.search_issues(experiment_id=exp_id2)

    assert len(result) == 0
    assert result.token is None


def test_search_issues_filter_by_status(store):
    exp_id = store.create_experiment("test")

    # Create issues with different statuses
    store.create_issue(
        experiment_id=exp_id,
        name="Draft Issue",
        description="In draft",
        status="draft",
    )

    issue_accepted = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Issue",
        description="Accepted",
        status="accepted",
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Rejected Issue",
        description="Rejected",
        status="rejected",
    )

    # Filter by status using filter_string
    result = store.search_issues(filter_string="status = 'accepted'")

    assert len(result) == 1
    assert result[0].issue_id == issue_accepted.issue_id
    assert result[0].status == "accepted"


def test_search_issues_filter_by_source_run_id(store):
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
        name="Run1 Issue",
        description="From run 1",
        status="draft",
        source_run_id=run1.info.run_id,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Run2 Issue",
        description="From run 2",
        status="draft",
        source_run_id=run2.info.run_id,
    )

    # Filter by source_run_id using filter_string
    result = store.search_issues(filter_string=f"source_run_id = '{run1.info.run_id}'")

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].source_run_id == run1.info.run_id


def test_search_issues_filter_combined_with_experiment_id(store):
    exp_id1 = store.create_experiment("test1")
    exp_id2 = store.create_experiment("test2")

    # Create issues in different experiments with different statuses
    issue1 = store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Draft",
        description="Draft in exp1",
        status="draft",
    )

    store.create_issue(
        experiment_id=exp_id1,
        name="Exp1 Accepted",
        description="Accepted in exp1",
        status="accepted",
    )

    store.create_issue(
        experiment_id=exp_id2,
        name="Exp2 Draft",
        description="Draft in exp2",
        status="draft",
    )

    # Filter by experiment_id and status
    result = store.search_issues(experiment_id=exp_id1, filter_string="status = 'draft'")

    assert len(result) == 1
    assert result[0].issue_id == issue1.issue_id
    assert result[0].experiment_id == exp_id1
    assert result[0].status == "draft"


def test_search_issues_filter_invalid_field(store):
    exp_id = store.create_experiment("test")

    store.create_issue(
        experiment_id=exp_id,
        name="Test Issue",
        description="Test",
        status="draft",
    )

    # Filter by invalid field should raise error
    with pytest.raises(MlflowException, match=r"Invalid filter field 'invalid_field'"):
        store.search_issues(filter_string="invalid_field = 'value'")


def test_search_issues_filter_inequality(store):
    exp_id = store.create_experiment("test")

    # Create issues with different statuses
    store.create_issue(
        experiment_id=exp_id,
        name="Draft Issue",
        description="In draft",
        status="draft",
    )

    issue_accepted = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Issue",
        description="Accepted",
        status="accepted",
    )

    # Filter by status != 'draft' should return non-draft issues
    result = store.search_issues(filter_string="status != 'draft'")

    assert len(result) == 1
    assert result[0].issue_id == issue_accepted.issue_id
    assert result[0].status == "accepted"


def test_search_issues_filter_and_operator(store):
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

    # Create issues with different combinations
    store.create_issue(
        experiment_id=exp_id,
        name="Draft Run1",
        description="Draft from run1",
        status="draft",
        source_run_id=run1.info.run_id,
    )

    issue_accepted_run1 = store.create_issue(
        experiment_id=exp_id,
        name="Accepted Run1",
        description="Accepted from run1",
        status="accepted",
        source_run_id=run1.info.run_id,
    )

    store.create_issue(
        experiment_id=exp_id,
        name="Accepted Run2",
        description="Accepted from run2",
        status="accepted",
        source_run_id=run2.info.run_id,
    )

    # Filter with AND: status = 'accepted' AND source_run_id = run1
    result = store.search_issues(
        filter_string=f"status = 'accepted' AND source_run_id = '{run1.info.run_id}'"
    )

    assert len(result) == 1
    assert result[0].issue_id == issue_accepted_run1.issue_id
    assert result[0].status == "accepted"
    assert result[0].source_run_id == run1.info.run_id
