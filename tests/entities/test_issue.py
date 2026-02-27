from mlflow.entities.issue import Issue
from mlflow.protos.issues_pb2 import Issue as ProtoIssue


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


def test_issue_to_proto_required_fields():
    issue = Issue(
        issue_id="iss-proto-1",
        experiment_id="exp-proto-1",
        name="Proto test",
        description="Testing proto conversion",
        frequency=0.8,
        status="draft",
        created_timestamp=1000000000,
        last_updated_timestamp=1000000001,
    )

    proto = issue.to_proto()

    assert proto.issue_id == "iss-proto-1"
    assert proto.experiment_id == "exp-proto-1"
    assert proto.name == "Proto test"
    assert proto.description == "Testing proto conversion"
    assert proto.frequency == 0.8
    assert proto.status == "draft"
    assert proto.created_timestamp == 1000000000
    assert proto.last_updated_timestamp == 1000000001
    assert proto.run_id == ""
    assert proto.root_cause == ""
    assert proto.confidence == ""
    assert len(proto.rationale_examples) == 0
    assert len(proto.example_trace_ids) == 0
    assert len(proto.trace_ids) == 0
    assert proto.created_by == ""


def test_issue_to_proto_all_fields():
    issue = Issue(
        issue_id="iss-proto-2",
        experiment_id="exp-proto-2",
        name="Full proto test",
        description="Testing proto conversion with all fields",
        frequency=0.95,
        status="accepted",
        created_timestamp=2000000000,
        last_updated_timestamp=2000000010,
        run_id="run-proto-2",
        root_cause="Proto test root cause",
        confidence="very_high",
        rationale_examples=["Proto example 1", "Proto example 2"],
        example_trace_ids=["proto-trace-1", "proto-trace-2"],
        trace_ids=["proto-trace-1", "proto-trace-2", "proto-trace-3"],
        created_by="proto-user@example.com",
    )

    proto = issue.to_proto()

    assert proto.issue_id == "iss-proto-2"
    assert proto.experiment_id == "exp-proto-2"
    assert proto.name == "Full proto test"
    assert proto.description == "Testing proto conversion with all fields"
    assert proto.frequency == 0.95
    assert proto.status == "accepted"
    assert proto.created_timestamp == 2000000000
    assert proto.last_updated_timestamp == 2000000010
    assert proto.run_id == "run-proto-2"
    assert proto.root_cause == "Proto test root cause"
    assert proto.confidence == "very_high"
    assert list(proto.rationale_examples) == ["Proto example 1", "Proto example 2"]
    assert list(proto.example_trace_ids) == ["proto-trace-1", "proto-trace-2"]
    assert list(proto.trace_ids) == ["proto-trace-1", "proto-trace-2", "proto-trace-3"]
    assert proto.created_by == "proto-user@example.com"


def test_issue_from_proto_required_fields():

    proto = ProtoIssue(
        issue_id="iss-from-proto-1",
        experiment_id="exp-from-proto-1",
        name="From proto test",
        description="Testing conversion from proto",
        frequency=0.6,
        status="draft",
        created_timestamp=3000000000,
        last_updated_timestamp=3000000001,
    )

    issue = Issue.from_proto(proto)

    assert issue.issue_id == "iss-from-proto-1"
    assert issue.experiment_id == "exp-from-proto-1"
    assert issue.name == "From proto test"
    assert issue.description == "Testing conversion from proto"
    assert issue.frequency == 0.6
    assert issue.status == "draft"
    assert issue.created_timestamp == 3000000000
    assert issue.last_updated_timestamp == 3000000001
    assert issue.run_id is None
    assert issue.root_cause is None
    assert issue.confidence is None
    assert issue.rationale_examples is None
    assert issue.example_trace_ids is None
    assert issue.trace_ids is None
    assert issue.created_by is None


def test_issue_from_proto_all_fields():

    proto = ProtoIssue(
        issue_id="iss-from-proto-2",
        experiment_id="exp-from-proto-2",
        name="Full from proto test",
        description="Testing conversion from proto with all fields",
        frequency=0.85,
        status="rejected",
        created_timestamp=4000000000,
        last_updated_timestamp=4000000020,
        run_id="run-from-proto-2",
        root_cause="From proto root cause",
        confidence="low",
        created_by="from-proto-user@example.com",
    )
    proto.rationale_examples.extend(["From proto example 1", "From proto example 2"])
    proto.example_trace_ids.extend(["from-proto-trace-1", "from-proto-trace-2"])
    proto.trace_ids.extend(["from-proto-trace-1", "from-proto-trace-2", "from-proto-trace-3"])

    issue = Issue.from_proto(proto)

    assert issue.issue_id == "iss-from-proto-2"
    assert issue.experiment_id == "exp-from-proto-2"
    assert issue.name == "Full from proto test"
    assert issue.description == "Testing conversion from proto with all fields"
    assert issue.frequency == 0.85
    assert issue.status == "rejected"
    assert issue.created_timestamp == 4000000000
    assert issue.last_updated_timestamp == 4000000020
    assert issue.run_id == "run-from-proto-2"
    assert issue.root_cause == "From proto root cause"
    assert issue.confidence == "low"
    assert issue.rationale_examples == ["From proto example 1", "From proto example 2"]
    assert issue.example_trace_ids == ["from-proto-trace-1", "from-proto-trace-2"]
    assert issue.trace_ids == [
        "from-proto-trace-1",
        "from-proto-trace-2",
        "from-proto-trace-3",
    ]
    assert issue.created_by == "from-proto-user@example.com"


def test_issue_proto_roundtrip_required_fields():
    original = Issue(
        issue_id="iss-proto-roundtrip-1",
        experiment_id="exp-proto-roundtrip-1",
        name="Proto roundtrip test",
        description="Testing proto roundtrip conversion",
        frequency=0.72,
        status="accepted",
        created_timestamp=5000000000,
        last_updated_timestamp=5000000005,
    )

    proto = original.to_proto()
    recovered = Issue.from_proto(proto)

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


def test_issue_proto_roundtrip_all_fields():
    original = Issue(
        issue_id="iss-proto-roundtrip-2",
        experiment_id="exp-proto-roundtrip-2",
        name="Full proto roundtrip test",
        description="Testing proto roundtrip with all fields",
        frequency=0.93,
        status="draft",
        created_timestamp=6000000000,
        last_updated_timestamp=6000000030,
        run_id="run-proto-roundtrip-2",
        root_cause="Proto roundtrip root cause",
        confidence="medium",
        rationale_examples=["Roundtrip example 1", "Roundtrip example 2", "Roundtrip example 3"],
        example_trace_ids=["roundtrip-trace-1", "roundtrip-trace-2"],
        trace_ids=["roundtrip-trace-1", "roundtrip-trace-2", "roundtrip-trace-3"],
        created_by="roundtrip-user@example.com",
    )

    proto = original.to_proto()
    recovered = Issue.from_proto(proto)

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
