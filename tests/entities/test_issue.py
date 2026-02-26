from mlflow.entities.issue import Issue


def test_issue_creation_required_fields():
    issue = Issue(
        issue_id="iss-123",
        experiment_id="exp-123",
        name="High latency",
        description="API calls are taking too long",
        frequency=0.75,
        status="draft",
        created_timestamp=1234567890,
        last_updated_timestamp=1234567890,
    )

    assert issue.issue_id == "iss-123"
    assert issue.experiment_id == "exp-123"
    assert issue.name == "High latency"
    assert issue.description == "API calls are taking too long"
    assert issue.frequency == 0.75
    assert issue.status == "draft"
    assert issue.created_timestamp == 1234567890
    assert issue.last_updated_timestamp == 1234567890
    assert issue.run_id is None
    assert issue.root_cause is None
    assert issue.confidence is None
    assert issue.rationale_examples is None
    assert issue.example_trace_ids is None
    assert issue.trace_ids is None
    assert issue.created_by is None


def test_issue_creation_all_fields():
    issue = Issue(
        issue_id="iss-456",
        experiment_id="exp-456",
        name="Token limit exceeded",
        description="Model is hitting token limits frequently",
        frequency=0.42,
        status="accepted",
        created_timestamp=1234567890,
        last_updated_timestamp=1234567900,
        run_id="run-789",
        root_cause="Input prompts are too long",
        confidence="high",
        rationale_examples=["Example 1", "Example 2"],
        example_trace_ids=["trace-1", "trace-2", "trace-3"],
        trace_ids=["trace-1", "trace-2", "trace-3", "trace-4", "trace-5"],
        created_by="user@example.com",
    )

    assert issue.issue_id == "iss-456"
    assert issue.experiment_id == "exp-456"
    assert issue.name == "Token limit exceeded"
    assert issue.description == "Model is hitting token limits frequently"
    assert issue.frequency == 0.42
    assert issue.status == "accepted"
    assert issue.created_timestamp == 1234567890
    assert issue.last_updated_timestamp == 1234567900
    assert issue.run_id == "run-789"
    assert issue.root_cause == "Input prompts are too long"
    assert issue.confidence == "high"
    assert issue.rationale_examples == ["Example 1", "Example 2"]
    assert issue.example_trace_ids == ["trace-1", "trace-2", "trace-3"]
    assert issue.trace_ids == ["trace-1", "trace-2", "trace-3", "trace-4", "trace-5"]
    assert issue.created_by == "user@example.com"


def test_issue_to_dictionary():
    issue = Issue(
        issue_id="iss-789",
        experiment_id="exp-789",
        name="Authentication failure",
        description="Users are getting auth errors",
        frequency=0.15,
        status="rejected",
        created_timestamp=9876543210,
        last_updated_timestamp=9876543220,
        run_id="run-abc",
        root_cause="API key rotation issue",
        confidence="medium",
        rationale_examples=["Error occurred 3 times"],
        example_trace_ids=["trace-x"],
        trace_ids=["trace-x", "trace-y"],
        created_by="system",
    )

    issue_dict = issue.to_dictionary()

    assert issue_dict["issue_id"] == "iss-789"
    assert issue_dict["experiment_id"] == "exp-789"
    assert issue_dict["name"] == "Authentication failure"
    assert issue_dict["description"] == "Users are getting auth errors"
    assert issue_dict["frequency"] == 0.15
    assert issue_dict["status"] == "rejected"
    assert issue_dict["created_timestamp"] == 9876543210
    assert issue_dict["last_updated_timestamp"] == 9876543220
    assert issue_dict["run_id"] == "run-abc"
    assert issue_dict["root_cause"] == "API key rotation issue"
    assert issue_dict["confidence"] == "medium"
    assert issue_dict["rationale_examples"] == ["Error occurred 3 times"]
    assert issue_dict["example_trace_ids"] == ["trace-x"]
    assert issue_dict["trace_ids"] == ["trace-x", "trace-y"]
    assert issue_dict["created_by"] == "system"


def test_issue_from_dictionary_all_fields():
    issue_dict = {
        "issue_id": "iss-999",
        "experiment_id": "exp-999",
        "run_id": "run-xyz",
        "name": "Low accuracy",
        "description": "Model accuracy below threshold",
        "root_cause": "Training data quality issues",
        "status": "draft",
        "frequency": 0.88,
        "confidence": "low",
        "rationale_examples": ["Rationale A", "Rationale B"],
        "example_trace_ids": ["trace-1", "trace-2"],
        "trace_ids": ["trace-1", "trace-2", "trace-3"],
        "created_timestamp": 1111111111,
        "last_updated_timestamp": 2222222222,
        "created_by": "admin@example.com",
    }

    issue = Issue.from_dictionary(issue_dict)

    assert issue.issue_id == "iss-999"
    assert issue.experiment_id == "exp-999"
    assert issue.run_id == "run-xyz"
    assert issue.name == "Low accuracy"
    assert issue.description == "Model accuracy below threshold"
    assert issue.root_cause == "Training data quality issues"
    assert issue.status == "draft"
    assert issue.frequency == 0.88
    assert issue.confidence == "low"
    assert issue.rationale_examples == ["Rationale A", "Rationale B"]
    assert issue.example_trace_ids == ["trace-1", "trace-2"]
    assert issue.trace_ids == ["trace-1", "trace-2", "trace-3"]
    assert issue.created_timestamp == 1111111111
    assert issue.last_updated_timestamp == 2222222222
    assert issue.created_by == "admin@example.com"


def test_issue_from_dictionary_required_fields_only():
    issue_dict = {
        "issue_id": "iss-minimal",
        "experiment_id": "exp-minimal",
        "name": "Minimal issue",
        "description": "Issue with only required fields",
        "status": "draft",
        "frequency": 0.5,
        "created_timestamp": 5555555555,
        "last_updated_timestamp": 5555555555,
    }

    issue = Issue.from_dictionary(issue_dict)

    assert issue.issue_id == "iss-minimal"
    assert issue.experiment_id == "exp-minimal"
    assert issue.name == "Minimal issue"
    assert issue.description == "Issue with only required fields"
    assert issue.status == "draft"
    assert issue.frequency == 0.5
    assert issue.created_timestamp == 5555555555
    assert issue.last_updated_timestamp == 5555555555
    assert issue.run_id is None
    assert issue.root_cause is None
    assert issue.confidence is None
    assert issue.rationale_examples is None
    assert issue.example_trace_ids is None
    assert issue.trace_ids is None
    assert issue.created_by is None


def test_issue_roundtrip_conversion():
    original = Issue(
        issue_id="iss-roundtrip",
        experiment_id="exp-roundtrip",
        name="Roundtrip test",
        description="Testing dictionary conversion",
        frequency=0.67,
        status="accepted",
        created_timestamp=3333333333,
        last_updated_timestamp=4444444444,
        run_id="run-test",
        root_cause="Test root cause",
        confidence="high",
        rationale_examples=["Test rationale"],
        example_trace_ids=["test-trace"],
        trace_ids=["test-trace", "test-trace-2"],
        created_by="test-user",
    )

    issue_dict = original.to_dictionary()
    recovered = Issue.from_dictionary(issue_dict)

    assert recovered.issue_id == original.issue_id
    assert recovered.experiment_id == original.experiment_id
    assert recovered.name == original.name
    assert recovered.description == original.description
    assert recovered.frequency == original.frequency
    assert recovered.status == original.status
    assert recovered.created_timestamp == original.created_timestamp
    assert recovered.last_updated_timestamp == original.last_updated_timestamp
    assert recovered.run_id == original.run_id
    assert recovered.root_cause == original.root_cause
    assert recovered.confidence == original.confidence
    assert recovered.rationale_examples == original.rationale_examples
    assert recovered.example_trace_ids == original.example_trace_ids
    assert recovered.trace_ids == original.trace_ids
    assert recovered.created_by == original.created_by
