from datetime import datetime
from time import sleep
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from mlflow.exceptions import MlflowException
from mlflow.insights.constants import (
    AnalysisStatus,
    HypothesisStatus,
    IssueSeverity,
    IssueStatus,
)
from mlflow.insights.models import (
    Analysis,
    AnalysisSummary,
    EvidenceEntry,
    Hypothesis,
    HypothesisSummary,
    Issue,
    IssueSummary,
)


@pytest.fixture
def analysis():
    return Analysis(name="Test", description="Test")


@pytest.fixture
def hypothesis():
    return Hypothesis(statement="Test", testing_plan="Test")


@pytest.fixture
def issue():
    return Issue(
        source_run_id="run123", title="Test", description="Test", severity=IssueSeverity.HIGH
    )


@pytest.fixture
def yaml_file(tmp_path):
    return tmp_path / "entity.yaml"


@pytest.mark.parametrize(
    (
        "trace_id",
        "rationale",
        "supports",
        "expected_trace",
        "expected_rationale",
        "expected_supports",
    ),
    [
        (
            "trace123",
            "This shows the model performing well",
            True,
            "trace123",
            "This shows the model performing well",
            True,
        ),
        (
            "trace456",
            "This trace demonstrates the error",
            None,
            "trace456",
            "This trace demonstrates the error",
            None,
        ),
        ("  trace_with_spaces  ", "test", None, "trace_with_spaces", "test", None),
        ("trace", "  rationale with spaces  ", None, "trace", "rationale with spaces", None),
    ],
)
def test_evidence_entry_creation(
    trace_id, rationale, supports, expected_trace, expected_rationale, expected_supports
):
    entry = EvidenceEntry(trace_id=trace_id, rationale=rationale, supports=supports)
    assert entry.trace_id == expected_trace
    assert entry.rationale == expected_rationale
    assert entry.supports == expected_supports


@pytest.mark.parametrize(
    ("method", "args", "expected_supports"),
    [
        ("for_hypothesis", ("trace789", "Supporting evidence", True), True),
        ("for_hypothesis", ("trace999", "Default support"), True),
        ("for_issue", ("issue_trace", "Issue evidence"), None),
    ],
)
def test_evidence_entry_factory_methods(method, args, expected_supports):
    factory_method = getattr(EvidenceEntry, method)
    entry = factory_method(*args)
    assert entry.trace_id == args[0]
    assert entry.rationale == args[1]
    assert entry.supports == expected_supports


@pytest.mark.parametrize("invalid_trace_id", ["", "   ", None])
def test_evidence_entry_empty_trace_id_raises(invalid_trace_id):
    with pytest.raises(
        (MlflowException, ValueError), match="trace_id|cannot be empty|none is not an allowed value"
    ):
        EvidenceEntry(trace_id=invalid_trace_id, rationale="valid rationale")


@pytest.mark.parametrize("invalid_rationale", ["", "   ", None])
def test_evidence_entry_empty_rationale_raises(invalid_rationale):
    with pytest.raises(
        (MlflowException, ValueError),
        match="rationale|cannot be empty|none is not an allowed value",
    ):
        EvidenceEntry(trace_id="valid_trace", rationale=invalid_rationale)


def test_analysis_creation_with_defaults(analysis):
    assert analysis.name == "Test"
    assert analysis.description == "Test"
    assert analysis.status == AnalysisStatus.ACTIVE
    assert isinstance(analysis.created_at, datetime)
    assert isinstance(analysis.updated_at, datetime)
    assert analysis.metadata == {}


@pytest.mark.parametrize(
    ("name", "description", "error_field"),
    [
        ("", "Valid description", "name"),
        ("   ", "Valid description", "name"),
        ("Valid name", "", "description"),
        ("Valid name", "   ", "description"),
    ],
)
def test_analysis_validation_errors(name, description, error_field):
    with pytest.raises(MlflowException, match=f"Analysis {error_field} cannot be empty"):
        Analysis(name=name, description=description)


@pytest.mark.parametrize(
    ("input_name", "input_desc", "expected_name", "expected_desc"),
    [
        ("  Test Analysis  ", "  Testing the model  ", "Test Analysis", "Testing the model"),
    ],
)
def test_analysis_strips_whitespace(input_name, input_desc, expected_name, expected_desc):
    analysis = Analysis(name=input_name, description=input_desc)
    assert analysis.name == expected_name
    assert analysis.description == expected_desc


@pytest.mark.parametrize(
    ("transitions", "expected_statuses"),
    [
        (
            ["complete", "archive", "reactivate"],
            [AnalysisStatus.COMPLETED, AnalysisStatus.ARCHIVED, AnalysisStatus.ACTIVE],
        ),
    ],
)
def test_analysis_status_transitions(analysis, transitions, expected_statuses):
    assert analysis.status == AnalysisStatus.ACTIVE
    previous_timestamp = analysis.updated_at

    for transition, expected_status in zip(transitions, expected_statuses):
        sleep(0.001)
        getattr(analysis, transition)()
        assert analysis.status == expected_status
        assert analysis.updated_at > previous_timestamp
        previous_timestamp = analysis.updated_at


@pytest.mark.parametrize(
    ("error_message", "should_have_message"),
    [
        (None, False),
        ("Something went wrong", True),
    ],
)
def test_analysis_mark_error(analysis, error_message, should_have_message):
    analysis.mark_error(error_message)
    assert analysis.status == AnalysisStatus.ERROR
    if should_have_message:
        assert analysis.metadata["error_message"] == error_message
    else:
        assert "error_message" not in analysis.metadata


def test_analysis_metadata_validation():
    analysis = Analysis(name="Test", description="Test", metadata={"key": "value", "count": 42})
    assert analysis.metadata == {"key": "value", "count": 42}

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        Analysis(name="Test", description="Test", metadata="not a dict")


def test_hypothesis_creation_with_defaults(hypothesis):
    assert hypothesis.statement == "Test"
    assert hypothesis.testing_plan == "Test"
    assert hypothesis.status == HypothesisStatus.TESTING
    assert UUID(hypothesis.hypothesis_id)
    assert hypothesis.evidence == []
    assert hypothesis.metrics == {}
    assert isinstance(hypothesis.created_at, datetime)


@pytest.mark.parametrize(
    ("statement", "testing_plan", "error_field"),
    [
        ("", "Valid plan", "statement"),
        ("   ", "Valid plan", "statement"),
        ("Valid statement", "", "testing plan"),
        ("Valid statement", "   ", "testing plan"),
    ],
)
def test_hypothesis_validation_errors(statement, testing_plan, error_field):
    with pytest.raises(MlflowException, match=f"Hypothesis {error_field} cannot be empty"):
        Hypothesis(statement=statement, testing_plan=testing_plan)


def test_hypothesis_evidence_normalization():
    hyp = Hypothesis(
        statement="Test",
        testing_plan="Test",
        evidence=[
            {"trace_id": "trace1", "rationale": "reason1", "supports": True},
            {"trace_id": "trace2", "rationale": "reason2", "supports": False},
        ],
    )
    assert len(hyp.evidence) == 2
    assert all(isinstance(e, EvidenceEntry) for e in hyp.evidence)
    assert hyp.evidence[0].supports is True
    assert hyp.evidence[1].supports is False

    evidence_entries = [
        EvidenceEntry.for_hypothesis("trace3", "reason3", True),
        EvidenceEntry.for_hypothesis("trace4", "reason4", False),
    ]
    hyp2 = Hypothesis(statement="Test", testing_plan="Test", evidence=evidence_entries)
    assert hyp2.evidence == evidence_entries


def test_hypothesis_evidence_counts():
    hyp = Hypothesis(
        statement="Test",
        testing_plan="Test",
        evidence=[
            {"trace_id": "trace1", "rationale": "supports", "supports": True},
            {"trace_id": "trace2", "rationale": "refutes", "supports": False},
            {"trace_id": "trace1", "rationale": "more support", "supports": True},
            {"trace_id": "trace3", "rationale": "neutral", "supports": True},
        ],
    )
    assert hyp.evidence_count == 4
    assert hyp.trace_count == 3
    assert hyp.supports_count == 3
    assert hyp.refutes_count == 1


def test_hypothesis_add_evidence(hypothesis):
    assert hypothesis.evidence_count == 0
    initial_timestamp = hypothesis.updated_at

    sleep(0.001)
    hypothesis.add_evidence("trace1", "Supporting evidence", supports=True)
    assert hypothesis.evidence_count == 1
    assert hypothesis.supports_count == 1
    assert hypothesis.updated_at > initial_timestamp

    sleep(0.001)
    hypothesis.add_evidence("trace2", "Refuting evidence", supports=False)
    assert hypothesis.evidence_count == 2
    assert hypothesis.refutes_count == 1


@pytest.mark.parametrize(
    ("transitions", "expected_statuses"),
    [
        (
            ["validate_hypothesis", "reopen_for_testing", "reject_hypothesis"],
            [HypothesisStatus.VALIDATED, HypothesisStatus.TESTING, HypothesisStatus.REJECTED],
        ),
    ],
)
def test_hypothesis_status_transitions(hypothesis, transitions, expected_statuses):
    assert hypothesis.status == HypothesisStatus.TESTING
    previous_timestamp = hypothesis.updated_at

    for transition, expected_status in zip(transitions, expected_statuses):
        sleep(0.001)
        getattr(hypothesis, transition)()
        assert hypothesis.status == expected_status
        assert hypothesis.updated_at > previous_timestamp
        previous_timestamp = hypothesis.updated_at


def test_hypothesis_mark_error(hypothesis):
    hypothesis.mark_error("Testing failed due to data issue")
    assert hypothesis.status == HypothesisStatus.ERROR
    assert hypothesis.metadata["error_message"] == "Testing failed due to data issue"


def test_hypothesis_metrics():
    hyp = Hypothesis(statement="Test", testing_plan="Test", metrics={"accuracy": 0.95, "f1": 0.92})
    assert hyp.metrics == {"accuracy": 0.95, "f1": 0.92}

    hyp.add_metric("precision", 0.94)
    assert hyp.metrics["precision"] == 0.94

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        Hypothesis(statement="Test", testing_plan="Test", metrics="not a dict")


def test_issue_creation_with_defaults(issue):
    assert issue.source_run_id == "run123"
    assert issue.title == "Test"
    assert issue.description == "Test"
    assert issue.severity == IssueSeverity.HIGH
    assert issue.status == IssueStatus.OPEN
    assert UUID(issue.issue_id)
    assert issue.hypothesis_id is None
    assert issue.assessments == []
    assert issue.resolution is None


@pytest.mark.parametrize(
    ("source_run_id", "title", "description", "error_field"),
    [
        ("", "Title", "Desc", "source_run_id"),
        ("run123", "", "Desc", "title"),
        ("run123", "Title", "", "description"),
    ],
)
def test_issue_validation_errors(source_run_id, title, description, error_field):
    with pytest.raises(MlflowException, match=f"Issue {error_field} cannot be empty"):
        Issue(
            source_run_id=source_run_id,
            title=title,
            description=description,
            severity=IssueSeverity.LOW,
        )


@pytest.mark.parametrize(
    (
        "input_run_id",
        "input_title",
        "input_desc",
        "expected_run_id",
        "expected_title",
        "expected_desc",
    ),
    [
        (
            "  run123  ",
            "  Performance Issue  ",
            "  Model accuracy dropped  ",
            "run123",
            "Performance Issue",
            "Model accuracy dropped",
        ),
    ],
)
def test_issue_strips_whitespace(
    input_run_id, input_title, input_desc, expected_run_id, expected_title, expected_desc
):
    issue = Issue(
        source_run_id=input_run_id,
        title=input_title,
        description=input_desc,
        severity=IssueSeverity.HIGH,
    )
    assert issue.source_run_id == expected_run_id
    assert issue.title == expected_title
    assert issue.description == expected_desc


def test_issue_evidence_normalization():
    issue = Issue(
        source_run_id="run123",
        title="Issue",
        description="Description",
        severity=IssueSeverity.MEDIUM,
        evidence=[
            {"trace_id": "trace1", "rationale": "Shows the problem"},
            {"trace_id": "trace2", "rationale": "Another example", "supports": True},
        ],
    )
    assert len(issue.evidence) == 2
    assert all(e.supports is None for e in issue.evidence)


def test_issue_add_evidence(issue):
    initial_timestamp = issue.updated_at

    sleep(0.001)
    issue.add_evidence("trace1", "Evidence of issue")
    assert issue.evidence_count == 1
    assert issue.evidence[0].supports is None
    assert issue.trace_count == 1
    assert issue.updated_at > initial_timestamp


def test_issue_assessments():
    issue = Issue(
        source_run_id="run123",
        title="Issue",
        description="Description",
        severity=IssueSeverity.HIGH,
        assessments=["  assessment1  ", "assessment2"],
    )
    assert issue.assessments == ["assessment1", "assessment2"]

    issue.add_assessment("assessment3")
    assert "assessment3" in issue.assessments

    issue.add_assessment("assessment1")
    assert issue.assessments.count("assessment1") == 1

    with pytest.raises(MlflowException, match="Assessment name cannot be empty"):
        issue.add_assessment("")


@pytest.mark.parametrize(
    ("transitions", "args", "expected_statuses", "expected_resolutions"),
    [
        (
            ["start_progress", "resolve", "reopen", "reject"],
            [None, "Fixed by updating config", None, "Not a real issue"],
            [IssueStatus.IN_PROGRESS, IssueStatus.RESOLVED, IssueStatus.OPEN, IssueStatus.REJECTED],
            [None, "Fixed by updating config", None, "Rejected: Not a real issue"],
        ),
    ],
)
def test_issue_status_transitions(
    issue, transitions, args, expected_statuses, expected_resolutions
):
    assert issue.status == IssueStatus.OPEN
    previous_timestamp = issue.updated_at

    for transition, arg, expected_status, expected_resolution in zip(
        transitions, args, expected_statuses, expected_resolutions
    ):
        sleep(0.001)
        method = getattr(issue, transition)
        method(arg) if arg else method()
        assert issue.status == expected_status
        assert issue.updated_at > previous_timestamp
        previous_timestamp = issue.updated_at

        if expected_resolution:
            assert expected_resolution in issue.resolution
        else:
            assert issue.resolution is None


def test_issue_mark_error(issue):
    issue.mark_error("Failed to process issue")
    assert issue.status == IssueStatus.ERROR
    assert issue.metadata["error_message"] == "Failed to process issue"


@pytest.mark.parametrize("resolution", ["", "   "])
def test_issue_resolve_validation(issue, resolution):
    with pytest.raises(MlflowException, match="Resolution description cannot be empty"):
        issue.resolve(resolution)


def test_analysis_summary_from_analysis(analysis):
    hypotheses = [
        Hypothesis(statement=f"Hyp {i}", testing_plan="Test", status=status)
        for i, status in enumerate(
            [
                HypothesisStatus.VALIDATED,
                HypothesisStatus.VALIDATED,
                HypothesisStatus.REJECTED,
                HypothesisStatus.TESTING,
            ]
        )
    ]

    summary = AnalysisSummary.from_analysis("run123", analysis, hypotheses)
    assert summary.run_id == "run123"
    assert summary.name == "Test"
    assert summary.description == "Test"
    assert summary.status == AnalysisStatus.ACTIVE
    assert summary.hypothesis_count == 4
    assert summary.validated_count == 2
    assert summary.rejected_count == 1
    assert summary.get_id() == "run123"


def test_hypothesis_summary_from_hypothesis():
    hyp = Hypothesis(
        statement="Test hypothesis",
        testing_plan="Test plan",
        status=HypothesisStatus.VALIDATED,
        evidence=[
            {"trace_id": "t1", "rationale": "r1", "supports": True},
            {"trace_id": "t2", "rationale": "r2", "supports": False},
            {"trace_id": "t1", "rationale": "r3", "supports": True},
        ],
    )

    summary = HypothesisSummary.from_hypothesis(hyp)
    assert summary.hypothesis_id == hyp.hypothesis_id
    assert summary.statement == "Test hypothesis"
    assert summary.status == HypothesisStatus.VALIDATED
    assert summary.trace_count == 2
    assert summary.evidence_count == 3
    assert summary.supports_count == 2
    assert summary.refutes_count == 1
    assert summary.get_id() == hyp.hypothesis_id


def test_issue_summary_from_issue():
    issue = Issue(
        source_run_id="run123",
        title="Critical Issue",
        description="Description",
        severity=IssueSeverity.CRITICAL,
        status=IssueStatus.RESOLVED,
        evidence=[
            {"trace_id": "t1", "rationale": "r1"},
            {"trace_id": "t2", "rationale": "r2"},
        ],
    )

    summary = IssueSummary.from_issue(issue)
    assert summary.issue_id == issue.issue_id
    assert summary.title == "Critical Issue"
    assert summary.severity == IssueSeverity.CRITICAL
    assert summary.status == IssueStatus.RESOLVED
    assert summary.trace_count == 2
    assert summary.source_run_id == "run123"
    assert summary.get_id() == issue.issue_id


@pytest.mark.parametrize(
    ("entity_fixture", "transitions"),
    [
        ("analysis", ["complete", "archive", "reactivate"]),
        ("hypothesis", ["validate_hypothesis", "reopen_for_testing", "reject_hypothesis"]),
    ],
)
def test_all_transitions_update_timestamp(request, entity_fixture, transitions):
    entity = request.getfixturevalue(entity_fixture)
    previous_timestamp = entity.updated_at

    for method_name in transitions:
        sleep(0.001)
        getattr(entity, method_name)()
        assert entity.updated_at > previous_timestamp
        previous_timestamp = entity.updated_at


def test_created_at_never_changes(analysis):
    initial_created = analysis.created_at

    sleep(0.001)
    analysis.complete()
    assert analysis.created_at == initial_created

    sleep(0.001)
    analysis.archive()
    assert analysis.created_at == initial_created


def test_analysis_serialization_lifecycle(yaml_file, analysis):
    initial_created = analysis.created_at
    initial_updated = analysis.updated_at

    with open(yaml_file, "w") as f:
        f.write(analysis.to_yaml())

    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    loaded = Analysis(**data)
    assert loaded.status == AnalysisStatus.ACTIVE
    assert loaded.created_at == initial_created
    assert loaded.updated_at == initial_updated

    sleep(0.001)
    loaded.complete()
    assert loaded.status == AnalysisStatus.COMPLETED
    assert loaded.updated_at > initial_updated
    assert loaded.created_at == initial_created

    with open(yaml_file, "w") as f:
        f.write(loaded.to_yaml())

    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    final = Analysis(**data)
    assert final.status == AnalysisStatus.COMPLETED
    assert final.created_at == initial_created
    assert final.updated_at == loaded.updated_at


def test_hypothesis_serialization_with_evidence(yaml_file, hypothesis):
    hypothesis_id = hypothesis.hypothesis_id

    hypothesis.add_evidence("trace1", "Evidence 1", supports=True)
    hypothesis.add_evidence("trace2", "Evidence 2", supports=False)

    with open(yaml_file, "w") as f:
        f.write(hypothesis.to_yaml())

    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    loaded = Hypothesis(**data)

    assert loaded.hypothesis_id == hypothesis_id
    assert loaded.evidence_count == 2
    assert loaded.supports_count == 1
    assert loaded.refutes_count == 1
    assert all(isinstance(e, EvidenceEntry) for e in loaded.evidence)


def test_issue_serialization_full_lifecycle(yaml_file, issue):
    issue_id = issue.issue_id

    issue.start_progress()
    issue.add_evidence("trace1", "Shows issue")
    issue.add_assessment("assessment1")

    with open(yaml_file, "w") as f:
        f.write(issue.to_yaml())

    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    loaded = Issue(**data)

    assert loaded.issue_id == issue_id
    assert loaded.status == IssueStatus.IN_PROGRESS
    assert loaded.evidence_count == 1
    assert "assessment1" in loaded.assessments

    sleep(0.001)
    loaded.resolve("Fixed")
    assert loaded.status == IssueStatus.RESOLVED
    assert loaded.resolution == "Fixed"

    with open(yaml_file, "w") as f:
        f.write(loaded.to_yaml())

    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    final = Issue(**data)
    assert final.status == IssueStatus.RESOLVED
    assert final.resolution == "Fixed"


@pytest.mark.parametrize(
    ("entity_class", "init_args"),
    [
        (Analysis, {"name": "Test", "description": "Test"}),
        (Hypothesis, {"statement": "Test", "testing_plan": "Test"}),
        (
            Issue,
            {
                "source_run_id": "run123",
                "title": "Test",
                "description": "Test",
                "severity": IssueSeverity.HIGH,
            },
        ),
    ],
)
def test_serialization_preserves_timestamps(yaml_file, entity_class, init_args):
    entity = entity_class(**init_args)
    original_created = entity.created_at
    original_updated = entity.updated_at

    with open(yaml_file, "w") as f:
        f.write(entity.to_yaml())

    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    loaded = entity_class(**data)

    assert loaded.created_at == original_created
    assert loaded.updated_at == original_updated
