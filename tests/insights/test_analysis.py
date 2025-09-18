import pytest
from datetime import datetime
from unittest.mock import patch
from uuid import UUID

from mlflow.exceptions import MlflowException
from mlflow.insights.analysis import (
    Analysis,
    AnalysisSummary,
    Hypothesis,
    HypothesisSummary,
    Issue,
    IssueSummary,
)
from mlflow.insights.constants import (
    AnalysisStatus,
    HypothesisStatus,
    IssueSeverity,
    IssueStatus,
)
from mlflow.insights.models import EvidenceEntry


class TestAnalysis:
    def test_analysis_creation_with_defaults(self):
        analysis = Analysis(
            name="Test Analysis",
            description="Testing the analysis model"
        )
        assert analysis.name == "Test Analysis"
        assert analysis.description == "Testing the analysis model"
        assert analysis.status == AnalysisStatus.ACTIVE
        assert isinstance(analysis.created_at, datetime)
        assert isinstance(analysis.updated_at, datetime)
        assert analysis.metadata == {}

    def test_analysis_name_validation(self):
        with pytest.raises(MlflowException, match="Analysis name cannot be empty"):
            Analysis(name="", description="Valid description")

        with pytest.raises(MlflowException, match="Analysis name cannot be empty"):
            Analysis(name="   ", description="Valid description")

    def test_analysis_description_validation(self):
        with pytest.raises(MlflowException, match="Analysis description cannot be empty"):
            Analysis(name="Valid name", description="")

        with pytest.raises(MlflowException, match="Analysis description cannot be empty"):
            Analysis(name="Valid name", description="   ")

    def test_analysis_strips_whitespace(self):
        analysis = Analysis(
            name="  Test Analysis  ",
            description="  Testing the model  "
        )
        assert analysis.name == "Test Analysis"
        assert analysis.description == "Testing the model"

    def test_analysis_status_transitions(self):
        analysis = Analysis(name="Test", description="Test")
        assert analysis.status == AnalysisStatus.ACTIVE

        initial_updated = analysis.updated_at
        with patch('mlflow.insights.analysis.datetime') as mock_dt:
            mock_dt.utcnow.return_value = datetime(2024, 1, 2)
            analysis.complete()
            assert analysis.status == AnalysisStatus.COMPLETED
            assert analysis.updated_at == datetime(2024, 1, 2)

            analysis.archive()
            assert analysis.status == AnalysisStatus.ARCHIVED

            analysis.reactivate()
            assert analysis.status == AnalysisStatus.ACTIVE

    def test_analysis_metadata_validation(self):
        analysis = Analysis(
            name="Test",
            description="Test",
            metadata={"key": "value", "count": 42}
        )
        assert analysis.metadata == {"key": "value", "count": 42}

        # Pydantic will raise ValidationError for non-dict types
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            Analysis(name="Test", description="Test", metadata="not a dict")


class TestHypothesis:
    def test_hypothesis_creation_with_defaults(self):
        hyp = Hypothesis(
            statement="Model performs better with feature X",
            testing_plan="Compare metrics with and without feature X"
        )
        assert hyp.statement == "Model performs better with feature X"
        assert hyp.testing_plan == "Compare metrics with and without feature X"
        assert hyp.status == HypothesisStatus.TESTING
        assert UUID(hyp.hypothesis_id)  # Validates it's a valid UUID
        assert hyp.evidence == []
        assert hyp.metrics == {}
        assert isinstance(hyp.created_at, datetime)

    def test_hypothesis_statement_validation(self):
        with pytest.raises(MlflowException, match="Hypothesis statement cannot be empty"):
            Hypothesis(statement="", testing_plan="Valid plan")

        with pytest.raises(MlflowException, match="Hypothesis statement cannot be empty"):
            Hypothesis(statement="   ", testing_plan="Valid plan")

    def test_hypothesis_testing_plan_validation(self):
        with pytest.raises(MlflowException, match="Hypothesis testing plan cannot be empty"):
            Hypothesis(statement="Valid statement", testing_plan="")

        with pytest.raises(MlflowException, match="Hypothesis testing plan cannot be empty"):
            Hypothesis(statement="Valid statement", testing_plan="   ")

    def test_hypothesis_evidence_normalization(self):
        # Test with dict evidence
        hyp = Hypothesis(
            statement="Test",
            testing_plan="Test",
            evidence=[
                {"trace_id": "trace1", "rationale": "reason1", "supports": True},
                {"trace_id": "trace2", "rationale": "reason2", "supports": False},
            ]
        )
        assert len(hyp.evidence) == 2
        assert all(isinstance(e, EvidenceEntry) for e in hyp.evidence)
        assert hyp.evidence[0].supports is True
        assert hyp.evidence[1].supports is False

        # Test with EvidenceEntry objects
        evidence_entries = [
            EvidenceEntry.for_hypothesis("trace3", "reason3", True),
            EvidenceEntry.for_hypothesis("trace4", "reason4", False),
        ]
        hyp2 = Hypothesis(
            statement="Test",
            testing_plan="Test",
            evidence=evidence_entries
        )
        assert hyp2.evidence == evidence_entries

    def test_hypothesis_evidence_counts(self):
        hyp = Hypothesis(
            statement="Test",
            testing_plan="Test",
            evidence=[
                {"trace_id": "trace1", "rationale": "supports", "supports": True},
                {"trace_id": "trace2", "rationale": "refutes", "supports": False},
                {"trace_id": "trace1", "rationale": "more support", "supports": True},
                {"trace_id": "trace3", "rationale": "neutral", "supports": True},
            ]
        )
        assert hyp.evidence_count == 4
        assert hyp.trace_count == 3  # trace1 appears twice
        assert hyp.supports_count == 3
        assert hyp.refutes_count == 1

    def test_hypothesis_add_evidence(self):
        hyp = Hypothesis(statement="Test", testing_plan="Test")
        assert hyp.evidence_count == 0

        with patch('mlflow.insights.analysis.datetime') as mock_dt:
            mock_dt.utcnow.return_value = datetime(2024, 1, 2)
            hyp.add_evidence("trace1", "Supporting evidence", supports=True)
            assert hyp.evidence_count == 1
            assert hyp.supports_count == 1
            assert hyp.updated_at == datetime(2024, 1, 2)

            hyp.add_evidence("trace2", "Refuting evidence", supports=False)
            assert hyp.evidence_count == 2
            assert hyp.refutes_count == 1

    def test_hypothesis_status_transitions(self):
        hyp = Hypothesis(statement="Test", testing_plan="Test")
        assert hyp.status == HypothesisStatus.TESTING

        hyp.validate_hypothesis()
        assert hyp.status == HypothesisStatus.VALIDATED

        hyp.reopen_for_testing()
        assert hyp.status == HypothesisStatus.TESTING

        hyp.reject_hypothesis()
        assert hyp.status == HypothesisStatus.REJECTED

    def test_hypothesis_metrics(self):
        hyp = Hypothesis(
            statement="Test",
            testing_plan="Test",
            metrics={"accuracy": 0.95, "f1": 0.92}
        )
        assert hyp.metrics == {"accuracy": 0.95, "f1": 0.92}

        hyp.add_metric("precision", 0.94)
        assert hyp.metrics["precision"] == 0.94

        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            Hypothesis(statement="Test", testing_plan="Test", metrics="not a dict")


class TestIssue:
    def test_issue_creation_with_defaults(self):
        issue = Issue(
            source_run_id="run123",
            title="Performance degradation",
            description="Model accuracy dropped by 10%",
            severity=IssueSeverity.HIGH
        )
        assert issue.source_run_id == "run123"
        assert issue.title == "Performance degradation"
        assert issue.description == "Model accuracy dropped by 10%"
        assert issue.severity == IssueSeverity.HIGH
        assert issue.status == IssueStatus.OPEN
        assert UUID(issue.issue_id)
        assert issue.hypothesis_id is None
        assert issue.assessments == []
        assert issue.resolution is None

    def test_issue_required_field_validation(self):
        with pytest.raises(MlflowException, match="Issue source_run_id cannot be empty"):
            Issue(
                source_run_id="",
                title="Title",
                description="Desc",
                severity=IssueSeverity.LOW
            )

        with pytest.raises(MlflowException, match="Issue title cannot be empty"):
            Issue(
                source_run_id="run123",
                title="",
                description="Desc",
                severity=IssueSeverity.LOW
            )

        with pytest.raises(MlflowException, match="Issue description cannot be empty"):
            Issue(
                source_run_id="run123",
                title="Title",
                description="",
                severity=IssueSeverity.LOW
            )

    def test_issue_strips_whitespace(self):
        issue = Issue(
            source_run_id="  run123  ",
            title="  Performance Issue  ",
            description="  Model accuracy dropped  ",
            severity=IssueSeverity.HIGH
        )
        assert issue.source_run_id == "run123"
        assert issue.title == "Performance Issue"
        assert issue.description == "Model accuracy dropped"

    def test_issue_evidence_normalization(self):
        # Issue evidence should always have supports=None
        issue = Issue(
            source_run_id="run123",
            title="Issue",
            description="Description",
            severity=IssueSeverity.MEDIUM,
            evidence=[
                {"trace_id": "trace1", "rationale": "Shows the problem"},
                {"trace_id": "trace2", "rationale": "Another example", "supports": True},
            ]
        )
        assert len(issue.evidence) == 2
        assert all(e.supports is None for e in issue.evidence)

    def test_issue_add_evidence(self):
        issue = Issue(
            source_run_id="run123",
            title="Issue",
            description="Description",
            severity=IssueSeverity.LOW
        )
        issue.add_evidence("trace1", "Evidence of issue")
        assert issue.evidence_count == 1
        assert issue.evidence[0].supports is None
        assert issue.trace_count == 1

    def test_issue_assessments(self):
        issue = Issue(
            source_run_id="run123",
            title="Issue",
            description="Description",
            severity=IssueSeverity.HIGH,
            assessments=["  assessment1  ", "assessment2"]
        )
        assert issue.assessments == ["assessment1", "assessment2"]

        issue.add_assessment("assessment3")
        assert "assessment3" in issue.assessments

        # Shouldn't add duplicates
        issue.add_assessment("assessment1")
        assert issue.assessments.count("assessment1") == 1

        # Empty assessments should be rejected
        with pytest.raises(MlflowException, match="Assessment name cannot be empty"):
            issue.add_assessment("")

    def test_issue_status_transitions(self):
        issue = Issue(
            source_run_id="run123",
            title="Issue",
            description="Description",
            severity=IssueSeverity.HIGH
        )
        assert issue.status == IssueStatus.OPEN

        issue.start_progress()
        assert issue.status == IssueStatus.IN_PROGRESS

        issue.resolve("Fixed by updating config")
        assert issue.status == IssueStatus.RESOLVED
        assert issue.resolution == "Fixed by updating config"

        issue.reopen()
        assert issue.status == IssueStatus.OPEN
        assert issue.resolution is None

        issue.reject("Not a real issue")
        assert issue.status == IssueStatus.REJECTED
        assert "Rejected: Not a real issue" in issue.resolution

    def test_issue_resolve_validation(self):
        issue = Issue(
            source_run_id="run123",
            title="Issue",
            description="Description",
            severity=IssueSeverity.HIGH
        )
        with pytest.raises(MlflowException, match="Resolution description cannot be empty"):
            issue.resolve("")

        with pytest.raises(MlflowException, match="Resolution description cannot be empty"):
            issue.resolve("   ")


class TestSummaryModels:
    def test_analysis_summary_from_analysis(self):
        analysis = Analysis(
            name="Test Analysis",
            description="Test Description",
            status=AnalysisStatus.COMPLETED
        )

        hypotheses = [
            Hypothesis(
                statement=f"Hyp {i}",
                testing_plan="Test",
                status=status
            )
            for i, status in enumerate([
                HypothesisStatus.VALIDATED,
                HypothesisStatus.VALIDATED,
                HypothesisStatus.REJECTED,
                HypothesisStatus.TESTING,
            ])
        ]

        summary = AnalysisSummary.from_analysis("run123", analysis, hypotheses)
        assert summary.run_id == "run123"
        assert summary.name == "Test Analysis"
        assert summary.description == "Test Description"
        assert summary.status == AnalysisStatus.COMPLETED
        assert summary.hypothesis_count == 4
        assert summary.validated_count == 2
        assert summary.rejected_count == 1
        assert summary.get_id() == "run123"

    def test_hypothesis_summary_from_hypothesis(self):
        hyp = Hypothesis(
            statement="Test hypothesis",
            testing_plan="Test plan",
            status=HypothesisStatus.VALIDATED,
            evidence=[
                {"trace_id": "t1", "rationale": "r1", "supports": True},
                {"trace_id": "t2", "rationale": "r2", "supports": False},
                {"trace_id": "t1", "rationale": "r3", "supports": True},
            ]
        )

        summary = HypothesisSummary.from_hypothesis(hyp)
        assert summary.hypothesis_id == hyp.hypothesis_id
        assert summary.statement == "Test hypothesis"
        assert summary.status == HypothesisStatus.VALIDATED
        assert summary.trace_count == 2  # t1 appears twice
        assert summary.evidence_count == 3
        assert summary.supports_count == 2
        assert summary.refutes_count == 1
        assert summary.get_id() == hyp.hypothesis_id

    def test_issue_summary_from_issue(self):
        issue = Issue(
            source_run_id="run123",
            title="Critical Issue",
            description="Description",
            severity=IssueSeverity.CRITICAL,
            status=IssueStatus.RESOLVED,
            evidence=[
                {"trace_id": "t1", "rationale": "r1"},
                {"trace_id": "t2", "rationale": "r2"},
            ]
        )

        summary = IssueSummary.from_issue(issue)
        assert summary.issue_id == issue.issue_id
        assert summary.title == "Critical Issue"
        assert summary.severity == IssueSeverity.CRITICAL
        assert summary.status == IssueStatus.RESOLVED
        assert summary.trace_count == 2
        assert summary.source_run_id == "run123"
        assert summary.get_id() == issue.issue_id