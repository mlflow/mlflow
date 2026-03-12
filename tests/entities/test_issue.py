import uuid

import pytest

from mlflow.entities.issue import Issue, IssueSeverity, IssueStatus
from mlflow.protos.issues_pb2 import Issue as ProtoIssue


def create_issue(severity: IssueSeverity) -> Issue:
    return Issue(
        issue_id="iss-" + str(uuid.uuid4()),
        experiment_id="exp-123",
        name="High latency",
        description="API calls are taking too long",
        status=IssueStatus.PENDING,
        created_timestamp=1234567890,
        last_updated_timestamp=1234567890,
        severity=severity,
    )


def test_issue_status_enum_values():
    assert IssueStatus.PENDING.value == "pending"
    assert IssueStatus.ACCEPTED.value == "accepted"
    assert IssueStatus.REJECTED.value == "rejected"
    assert IssueStatus.RESOLVED.value == "resolved"


def test_issue_status_enum_string_behavior():
    assert IssueStatus.PENDING == "pending"
    assert IssueStatus.ACCEPTED == "accepted"
    assert IssueStatus.REJECTED == "rejected"
    assert IssueStatus.RESOLVED == "resolved"


def test_issue_status_enum_from_string():
    assert IssueStatus("pending") == IssueStatus.PENDING
    assert IssueStatus("accepted") == IssueStatus.ACCEPTED
    assert IssueStatus("rejected") == IssueStatus.REJECTED
    assert IssueStatus("resolved") == IssueStatus.RESOLVED


def test_issue_status_enum_invalid_value():
    with pytest.raises(ValueError, match="'invalid' is not a valid IssueStatus"):
        IssueStatus("invalid")


def test_issue_status_enum_str_method():
    assert str(IssueStatus.PENDING) == "pending"
    assert str(IssueStatus.ACCEPTED) == "accepted"
    assert str(IssueStatus.REJECTED) == "rejected"
    assert str(IssueStatus.RESOLVED) == "resolved"


def test_issue_status_enum_isinstance():
    assert isinstance(IssueStatus.PENDING, str)
    assert isinstance(IssueStatus.PENDING, IssueStatus)


def test_issue_severity_enum_values():
    assert IssueSeverity.NOT_AN_ISSUE.value == "not_an_issue"
    assert IssueSeverity.LOW.value == "low"
    assert IssueSeverity.MEDIUM.value == "medium"
    assert IssueSeverity.HIGH.value == "high"


def test_issue_severity_enum_string_behavior():
    assert IssueSeverity.NOT_AN_ISSUE == "not_an_issue"
    assert IssueSeverity.LOW == "low"
    assert IssueSeverity.MEDIUM == "medium"
    assert IssueSeverity.HIGH == "high"


def test_issue_severity_enum_from_string():
    assert IssueSeverity("not_an_issue") == IssueSeverity.NOT_AN_ISSUE
    assert IssueSeverity("low") == IssueSeverity.LOW
    assert IssueSeverity("medium") == IssueSeverity.MEDIUM
    assert IssueSeverity("high") == IssueSeverity.HIGH


def test_issue_severity_enum_invalid_value():
    with pytest.raises(ValueError, match="'invalid' is not a valid IssueSeverity"):
        IssueSeverity("invalid")


def test_issue_severity_enum_str_method():
    assert str(IssueSeverity.NOT_AN_ISSUE) == "not_an_issue"
    assert str(IssueSeverity.LOW) == "low"
    assert str(IssueSeverity.MEDIUM) == "medium"
    assert str(IssueSeverity.HIGH) == "high"


def test_issue_severity_enum_isinstance():
    assert isinstance(IssueSeverity.LOW, str)
    assert isinstance(IssueSeverity.LOW, IssueSeverity)


def test_issue_severity_enum_comparison():
    assert IssueSeverity.LOW < IssueSeverity.MEDIUM
    assert IssueSeverity.MEDIUM < IssueSeverity.HIGH
    assert IssueSeverity.NOT_AN_ISSUE < IssueSeverity.LOW
    assert IssueSeverity.HIGH > IssueSeverity.MEDIUM
    assert IssueSeverity.MEDIUM > IssueSeverity.LOW
    assert IssueSeverity.LOW > IssueSeverity.NOT_AN_ISSUE
    assert IssueSeverity.LOW <= IssueSeverity.LOW
    assert IssueSeverity.LOW >= IssueSeverity.LOW
    assert IssueSeverity.HIGH >= IssueSeverity.MEDIUM
    assert IssueSeverity.MEDIUM <= IssueSeverity.HIGH


def test_issue_severity_enum_max():
    assert max(IssueSeverity.LOW, IssueSeverity.HIGH) == IssueSeverity.HIGH
    assert max(IssueSeverity.HIGH, IssueSeverity.LOW) == IssueSeverity.HIGH
    assert max(IssueSeverity.MEDIUM, IssueSeverity.MEDIUM) == IssueSeverity.MEDIUM
    assert max(IssueSeverity.NOT_AN_ISSUE, IssueSeverity.LOW) == IssueSeverity.LOW


def test_issue_sort_by_severity():
    issues = [
        create_issue(IssueSeverity.LOW),
        create_issue(IssueSeverity.NOT_AN_ISSUE),
        create_issue(IssueSeverity.HIGH),
        create_issue(IssueSeverity.MEDIUM),
        create_issue(IssueSeverity.NOT_AN_ISSUE),
    ]
    issues.sort(key=lambda i: i.severity, reverse=True)
    assert [issue.severity for issue in issues] == [
        IssueSeverity.HIGH,
        IssueSeverity.MEDIUM,
        IssueSeverity.LOW,
        IssueSeverity.NOT_AN_ISSUE,
        IssueSeverity.NOT_AN_ISSUE,
    ]


def test_issue_creation_required_fields():
    issue = Issue(
        issue_id="iss-123",
        experiment_id="exp-123",
        name="High latency",
        description="API calls are taking too long",
        status=IssueStatus.PENDING,
        created_timestamp=1234567890,
        last_updated_timestamp=1234567890,
    )

    assert issue.issue_id == "iss-123"
    assert issue.experiment_id == "exp-123"
    assert issue.name == "High latency"
    assert issue.description == "API calls are taking too long"
    assert issue.status == IssueStatus.PENDING
    assert issue.created_timestamp == 1234567890
    assert issue.last_updated_timestamp == 1234567890
    assert issue.severity is None
    assert issue.root_causes is None
    assert issue.source_run_id is None
    assert issue.created_by is None


def test_issue_creation_all_fields():
    issue = Issue(
        issue_id="iss-456",
        experiment_id="exp-456",
        name="Token limit exceeded",
        description="Model is hitting token limits frequently",
        status=IssueStatus.ACCEPTED,
        created_timestamp=1234567890,
        last_updated_timestamp=1234567900,
        severity=IssueSeverity.HIGH,
        root_causes=["Input prompts are too long", "Context window exceeded"],
        source_run_id="run-789",
        created_by="user@example.com",
    )

    assert issue.issue_id == "iss-456"
    assert issue.experiment_id == "exp-456"
    assert issue.name == "Token limit exceeded"
    assert issue.description == "Model is hitting token limits frequently"
    assert issue.status == IssueStatus.ACCEPTED
    assert issue.created_timestamp == 1234567890
    assert issue.last_updated_timestamp == 1234567900
    assert issue.severity == IssueSeverity.HIGH
    assert issue.root_causes == ["Input prompts are too long", "Context window exceeded"]
    assert issue.source_run_id == "run-789"
    assert issue.created_by == "user@example.com"


def test_issue_to_dictionary():
    issue = Issue(
        issue_id="iss-789",
        experiment_id="exp-789",
        name="Authentication failure",
        description="Users are getting auth errors",
        status=IssueStatus.REJECTED,
        created_timestamp=9876543210,
        last_updated_timestamp=9876543220,
        severity=IssueSeverity.MEDIUM,
        root_causes=["API key rotation issue", "Token expired"],
        source_run_id="run-abc",
        created_by="system",
    )

    issue_dict = issue.to_dictionary()

    assert issue_dict["issue_id"] == "iss-789"
    assert issue_dict["experiment_id"] == "exp-789"
    assert issue_dict["name"] == "Authentication failure"
    assert issue_dict["description"] == "Users are getting auth errors"
    assert issue_dict["status"] == "rejected"
    assert issue_dict["created_timestamp"] == 9876543210
    assert issue_dict["last_updated_timestamp"] == 9876543220
    assert issue_dict["severity"] == "medium"
    assert issue_dict["root_causes"] == ["API key rotation issue", "Token expired"]
    assert issue_dict["source_run_id"] == "run-abc"
    assert issue_dict["created_by"] == "system"


def test_issue_from_dictionary_all_fields():
    issue_dict = {
        "issue_id": "iss-999",
        "experiment_id": "exp-999",
        "name": "Low accuracy",
        "description": "Model accuracy below threshold",
        "status": "pending",
        "severity": "low",
        "root_causes": ["Training data quality issues", "Model drift"],
        "source_run_id": "run-xyz",
        "created_timestamp": 1111111111,
        "last_updated_timestamp": 2222222222,
        "created_by": "admin@example.com",
    }

    issue = Issue.from_dictionary(issue_dict)

    assert issue.issue_id == "iss-999"
    assert issue.experiment_id == "exp-999"
    assert issue.name == "Low accuracy"
    assert issue.description == "Model accuracy below threshold"
    assert issue.status == IssueStatus.PENDING
    assert issue.severity == IssueSeverity.LOW
    assert issue.root_causes == ["Training data quality issues", "Model drift"]
    assert issue.source_run_id == "run-xyz"
    assert issue.created_timestamp == 1111111111
    assert issue.last_updated_timestamp == 2222222222
    assert issue.created_by == "admin@example.com"


def test_issue_from_dictionary_required_fields_only():
    issue_dict = {
        "issue_id": "iss-minimal",
        "experiment_id": "exp-minimal",
        "name": "Minimal issue",
        "description": "Issue with only required fields",
        "status": "pending",
        "created_timestamp": 5555555555,
        "last_updated_timestamp": 5555555555,
    }

    issue = Issue.from_dictionary(issue_dict)

    assert issue.issue_id == "iss-minimal"
    assert issue.experiment_id == "exp-minimal"
    assert issue.name == "Minimal issue"
    assert issue.description == "Issue with only required fields"
    assert issue.status == IssueStatus.PENDING
    assert issue.created_timestamp == 5555555555
    assert issue.last_updated_timestamp == 5555555555
    assert issue.severity is None
    assert issue.root_causes is None
    assert issue.source_run_id is None
    assert issue.created_by is None


def test_issue_roundtrip_conversion():
    original = Issue(
        issue_id="iss-roundtrip",
        experiment_id="exp-roundtrip",
        name="Roundtrip test",
        description="Testing dictionary conversion",
        status=IssueStatus.ACCEPTED,
        created_timestamp=3333333333,
        last_updated_timestamp=4444444444,
        severity=IssueSeverity.HIGH,
        root_causes=["Test root cause", "Another cause"],
        source_run_id="run-test",
        created_by="test-user",
    )

    issue_dict = original.to_dictionary()
    recovered = Issue.from_dictionary(issue_dict)

    assert recovered.issue_id == original.issue_id
    assert recovered.experiment_id == original.experiment_id
    assert recovered.name == original.name
    assert recovered.description == original.description
    assert recovered.status == original.status
    assert recovered.created_timestamp == original.created_timestamp
    assert recovered.last_updated_timestamp == original.last_updated_timestamp
    assert recovered.severity == original.severity
    assert recovered.root_causes == original.root_causes
    assert recovered.source_run_id == original.source_run_id
    assert recovered.created_by == original.created_by


def test_issue_to_proto_required_fields():
    issue = Issue(
        issue_id="iss-proto-1",
        experiment_id="exp-proto-1",
        name="Proto test",
        description="Testing proto conversion",
        status=IssueStatus.PENDING,
        created_timestamp=1000000000,
        last_updated_timestamp=1000000001,
    )

    proto = issue.to_proto()

    assert proto.issue_id == "iss-proto-1"
    assert proto.experiment_id == "exp-proto-1"
    assert proto.name == "Proto test"
    assert proto.description == "Testing proto conversion"
    assert proto.status == "pending"
    assert proto.created_timestamp == 1000000000
    assert proto.last_updated_timestamp == 1000000001
    assert proto.severity == ""
    assert len(proto.root_causes) == 0
    assert proto.source_run_id == ""
    assert proto.created_by == ""


def test_issue_to_proto_all_fields():
    issue = Issue(
        issue_id="iss-proto-2",
        experiment_id="exp-proto-2",
        name="Full proto test",
        description="Testing proto conversion with all fields",
        status=IssueStatus.ACCEPTED,
        created_timestamp=2000000000,
        last_updated_timestamp=2000000010,
        severity=IssueSeverity.HIGH,
        root_causes=["Proto test root cause", "Another root cause"],
        source_run_id="run-proto-2",
        created_by="proto-user@example.com",
    )

    proto = issue.to_proto()

    assert proto.issue_id == "iss-proto-2"
    assert proto.experiment_id == "exp-proto-2"
    assert proto.name == "Full proto test"
    assert proto.description == "Testing proto conversion with all fields"
    assert proto.status == "accepted"
    assert proto.created_timestamp == 2000000000
    assert proto.last_updated_timestamp == 2000000010
    assert proto.severity == "high"
    assert list(proto.root_causes) == ["Proto test root cause", "Another root cause"]
    assert proto.source_run_id == "run-proto-2"
    assert proto.created_by == "proto-user@example.com"


def test_issue_from_proto_required_fields():
    proto = ProtoIssue(
        issue_id="iss-from-proto-1",
        experiment_id="exp-from-proto-1",
        name="From proto test",
        description="Testing conversion from proto",
        status="pending",
        created_timestamp=3000000000,
        last_updated_timestamp=3000000001,
    )

    issue = Issue.from_proto(proto)

    assert issue.issue_id == "iss-from-proto-1"
    assert issue.experiment_id == "exp-from-proto-1"
    assert issue.name == "From proto test"
    assert issue.description == "Testing conversion from proto"
    assert issue.status == IssueStatus.PENDING
    assert issue.created_timestamp == 3000000000
    assert issue.last_updated_timestamp == 3000000001
    assert issue.severity is None
    assert issue.root_causes is None
    assert issue.source_run_id is None
    assert issue.created_by is None


def test_issue_from_proto_all_fields():
    proto = ProtoIssue(
        issue_id="iss-from-proto-2",
        experiment_id="exp-from-proto-2",
        name="Full from proto test",
        description="Testing conversion from proto with all fields",
        status="rejected",
        created_timestamp=4000000000,
        last_updated_timestamp=4000000020,
        severity="low",
        source_run_id="run-from-proto-2",
        created_by="from-proto-user@example.com",
    )
    proto.root_causes.extend(["From proto root cause", "Another cause"])

    issue = Issue.from_proto(proto)

    assert issue.issue_id == "iss-from-proto-2"
    assert issue.experiment_id == "exp-from-proto-2"
    assert issue.name == "Full from proto test"
    assert issue.description == "Testing conversion from proto with all fields"
    assert issue.status == IssueStatus.REJECTED
    assert issue.created_timestamp == 4000000000
    assert issue.last_updated_timestamp == 4000000020
    assert issue.severity == IssueSeverity.LOW
    assert issue.root_causes == ["From proto root cause", "Another cause"]
    assert issue.source_run_id == "run-from-proto-2"
    assert issue.created_by == "from-proto-user@example.com"


def test_issue_proto_roundtrip_required_fields():
    original = Issue(
        issue_id="iss-proto-roundtrip-1",
        experiment_id="exp-proto-roundtrip-1",
        name="Proto roundtrip test",
        description="Testing proto roundtrip conversion",
        status=IssueStatus.ACCEPTED,
        created_timestamp=5000000000,
        last_updated_timestamp=5000000005,
    )

    proto = original.to_proto()
    recovered = Issue.from_proto(proto)

    assert recovered.issue_id == original.issue_id
    assert recovered.experiment_id == original.experiment_id
    assert recovered.name == original.name
    assert recovered.description == original.description
    assert recovered.status == original.status
    assert recovered.created_timestamp == original.created_timestamp
    assert recovered.last_updated_timestamp == original.last_updated_timestamp
    assert recovered.severity == original.severity
    assert recovered.root_causes == original.root_causes
    assert recovered.source_run_id == original.source_run_id
    assert recovered.created_by == original.created_by


def test_issue_proto_roundtrip_all_fields():
    original = Issue(
        issue_id="iss-proto-roundtrip-2",
        experiment_id="exp-proto-roundtrip-2",
        name="Full proto roundtrip test",
        description="Testing proto roundtrip with all fields",
        status=IssueStatus.PENDING,
        created_timestamp=6000000000,
        last_updated_timestamp=6000000030,
        severity=IssueSeverity.MEDIUM,
        root_causes=["Proto roundtrip root cause", "Secondary cause", "Tertiary cause"],
        source_run_id="run-proto-roundtrip-2",
        created_by="roundtrip-user@example.com",
    )

    proto = original.to_proto()
    recovered = Issue.from_proto(proto)

    assert recovered.issue_id == original.issue_id
    assert recovered.experiment_id == original.experiment_id
    assert recovered.name == original.name
    assert recovered.description == original.description
    assert recovered.status == original.status
    assert recovered.created_timestamp == original.created_timestamp
    assert recovered.last_updated_timestamp == original.last_updated_timestamp
    assert recovered.severity == original.severity
    assert recovered.root_causes == original.root_causes
    assert recovered.source_run_id == original.source_run_id
    assert recovered.created_by == original.created_by
