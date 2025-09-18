import pytest

from mlflow.exceptions import MlflowException
from mlflow.insights.models import EvidenceEntry


def test_evidence_entry_hypothesis_with_all_fields():
    entry = EvidenceEntry(
        trace_id="trace123", rationale="This shows the model performing well", supports=True
    )
    assert entry.trace_id == "trace123"
    assert entry.rationale == "This shows the model performing well"
    assert entry.supports is True


def test_evidence_entry_issue_without_supports():
    entry = EvidenceEntry(
        trace_id="trace456", rationale="This trace demonstrates the error", supports=None
    )
    assert entry.trace_id == "trace456"
    assert entry.rationale == "This trace demonstrates the error"
    assert entry.supports is None


def test_evidence_entry_for_hypothesis_method():
    entry = EvidenceEntry.for_hypothesis(
        trace_id="trace789", rationale="Supporting evidence", supports=True
    )
    assert entry.trace_id == "trace789"
    assert entry.rationale == "Supporting evidence"
    assert entry.supports is True


def test_evidence_entry_for_hypothesis_defaults_supports():
    entry = EvidenceEntry.for_hypothesis(trace_id="trace999", rationale="Default support")
    assert entry.supports is True


def test_evidence_entry_for_issue_method():
    entry = EvidenceEntry.for_issue(trace_id="issue_trace", rationale="Issue evidence")
    assert entry.trace_id == "issue_trace"
    assert entry.rationale == "Issue evidence"
    assert entry.supports is None


def test_evidence_entry_strips_trace_id_whitespace():
    entry = EvidenceEntry(trace_id="  trace_with_spaces  ", rationale="test")
    assert entry.trace_id == "trace_with_spaces"


def test_evidence_entry_strips_rationale_whitespace():
    entry = EvidenceEntry(trace_id="trace", rationale="  rationale with spaces  ")
    assert entry.rationale == "rationale with spaces"


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


def test_evidence_entry_supports_optional():
    entry = EvidenceEntry(trace_id="trace", rationale="test rationale")
    assert entry.supports is None
