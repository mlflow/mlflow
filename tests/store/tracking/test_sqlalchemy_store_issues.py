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
