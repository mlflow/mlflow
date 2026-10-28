import json

import pytest

from mlflow.entities.assessment import Feedback, IssueReference
from mlflow.entities.issue_evidence import (
    ISSUE_EVIDENCE_METADATA_KEY,
    MAX_ASSESSMENT_METADATA_BYTES,
    IssueEvidence,
    IssueEvidenceStep,
)
from mlflow.exceptions import MlflowException


def _step(**overrides):
    return {
        "span_id": "span-1",
        "text": "The retriever returned 0 documents",
        "highlights": ["0 documents"],
        "highlight_tone": "error",
        **overrides,
    }


def test_evidence_round_trips_through_metadata():
    evidence = IssueEvidence(
        summary="The agent answered from an empty context",
        steps=[
            _step(),
            _step(
                span_id="span-2", text="The model answered anyway", highlights=["anyway"], mono=True
            ),
        ],
        primary_span_id="span-2",
    )

    payload = json.loads(evidence.to_metadata_value())
    assert payload["schema_version"] == 1
    assert [step["span_id"] for step in payload["steps"]] == ["span-1", "span-2"]
    assert payload["steps"][1]["mono"] is True
    # An unset `mono` stays out of the payload rather than storing a default on every step.
    assert "mono" not in payload["steps"][0]
    assert evidence.span_ids == ["span-1", "span-2"]

    restored = IssueEvidence.from_metadata_value(evidence.summary, evidence.to_metadata_value())
    assert restored.summary == evidence.summary
    assert restored.steps == evidence.steps


def test_repeated_span_appears_once_in_span_ids():
    evidence = IssueEvidence(
        summary="One span, two observations",
        steps=[_step(), _step(text="It then retried", highlights=["retried"])],
    )
    assert evidence.span_ids == ["span-1"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"summary": "", "steps": [_step()]}, "summary must be a non-empty string"),
        ({"summary": "s", "steps": []}, "at least one step"),
        ({"summary": "s", "steps": [_step(span_id="  ")]}, "span_id must be a non-empty string"),
        ({"summary": "s", "steps": [_step(text="")]}, "text must be a non-empty string"),
        ({"summary": "s", "steps": [_step(highlights=[])]}, "at least one highlight"),
        ({"summary": "s", "steps": [_step(highlights=["absent"])]}, "exact substrings of text"),
        (
            {"summary": "s", "steps": [_step(highlight_tone="critical")]},
            "highlight_tone must be one of",
        ),
        ({"summary": "s", "steps": [_step(mono="yes")]}, "mono must be a boolean"),
        (
            {"summary": "s", "steps": [_step()], "primary_span_id": "span-9"},
            "must be cited by an evidence step",
        ),
        ({"summary": "s", "steps": ["not-an-object"]}, "steps\\[0\\] must be an object"),
    ],
)
def test_invalid_evidence_is_rejected(kwargs, message):
    with pytest.raises(MlflowException, match=message):
        IssueEvidence(**kwargs)


@pytest.mark.parametrize("tone", ["error", "warning", "success"])
def test_supported_tones(tone):
    assert IssueEvidenceStep(**_step(highlight_tone=tone)).highlight_tone == tone


def test_validate_spans_rejects_a_span_outside_the_trace():
    evidence = IssueEvidence(
        summary="s", steps=[_step(), _step(span_id="span-2", text="t", highlights=["t"])]
    )

    evidence.validate_spans(["span-1", "span-2", "span-3"])

    with pytest.raises(MlflowException, match="'span-2', which is not part of the trace"):
        evidence.validate_spans(["span-1"])


def test_metadata_is_bounded_by_what_the_store_writes():
    # Quotes double in size once the outer metadata document escapes them, so a payload that fits
    # on its own can still overflow the column.
    text = '"' * (MAX_ASSESSMENT_METADATA_BYTES // 3)
    evidence = IssueEvidence(summary="s", steps=[_step(text=text, highlights=['"'])])

    assert len(evidence.to_metadata_value().encode("utf-8")) < MAX_ASSESSMENT_METADATA_BYTES
    with pytest.raises(MlflowException, match="exceeds the 65535-byte limit"):
        evidence.to_metadata()


def test_caller_metadata_counts_against_the_same_budget():
    text = "x" * (MAX_ASSESSMENT_METADATA_BYTES - 200)
    evidence = IssueEvidence(summary="s", steps=[_step(text=text, highlights=["x"])])

    assert ISSUE_EVIDENCE_METADATA_KEY in evidence.to_metadata()
    with pytest.raises(MlflowException, match="exceeds the 65535-byte limit"):
        evidence.to_metadata({"detector": "x" * 300})


def test_metadata_merges_without_dropping_caller_keys():
    evidence = IssueEvidence(summary="s", steps=[_step()])

    merged = evidence.to_metadata({"detector": "offline"})

    assert merged["detector"] == "offline"
    assert merged[ISSUE_EVIDENCE_METADATA_KEY] == evidence.to_metadata_value()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[]", "must be an object"),
        (
            json.dumps({"schema_version": 2, "steps": [_step()]}),
            "Unsupported issue evidence schema version",
        ),
        (json.dumps({"steps": [_step()]}), "Unsupported issue evidence schema version"),
        (json.dumps({"schema_version": 1, "steps": []}), "at least one step"),
    ],
)
def test_malformed_metadata_is_rejected_on_an_explicit_read(value, message):
    with pytest.raises(MlflowException, match=message):
        IssueEvidence.from_metadata_value("summary", value)


def test_absent_metadata_reads_as_no_evidence():
    assert IssueEvidence.from_metadata_value("summary", None) is None
    assert IssueEvidence.from_metadata_value("summary", "") is None


def test_from_assessment_reads_structured_evidence():
    evidence = IssueEvidence(summary="Retrieved nothing", steps=[_step()], primary_span_id="span-1")
    assessment = IssueReference(
        issue_id="iss-1",
        issue_name="Empty retrieval",
        trace_id="tr-1",
        span_id="span-1",
        rationale=evidence.summary,
        metadata={ISSUE_EVIDENCE_METADATA_KEY: evidence.to_metadata_value()},
    )

    restored = IssueEvidence.from_assessment(assessment)
    assert restored.summary == "Retrieved nothing"
    assert restored.primary_span_id == "span-1"
    assert restored.steps == evidence.steps


def test_from_assessment_tolerates_legacy_and_malformed_references():
    legacy = IssueReference(
        issue_id="iss-1",
        issue_name="Empty retrieval",
        trace_id="tr-1",
        rationale="Only a rationale",
    )
    assert IssueEvidence.from_assessment(legacy) is None

    malformed = IssueReference(
        issue_id="iss-1",
        issue_name="Empty retrieval",
        trace_id="tr-1",
        rationale="Still readable",
        metadata={ISSUE_EVIDENCE_METADATA_KEY: "{broken"},
    )
    assert IssueEvidence.from_assessment(malformed) is None

    no_rationale = IssueReference(issue_id="iss-1", issue_name="Empty retrieval", trace_id="tr-1")
    assert IssueEvidence.from_assessment(no_rationale) is None


def test_from_assessment_ignores_a_primary_span_no_step_explains():
    evidence = IssueEvidence(summary="Retrieved nothing", steps=[_step()])
    assessment = IssueReference(
        issue_id="iss-1",
        issue_name="Empty retrieval",
        trace_id="tr-1",
        span_id="span-unexplained",
        rationale=evidence.summary,
        metadata={ISSUE_EVIDENCE_METADATA_KEY: evidence.to_metadata_value()},
    )

    assert IssueEvidence.from_assessment(assessment).primary_span_id is None


def test_from_assessment_ignores_unrelated_feedback():
    feedback = Feedback(name="quality", value=True, trace_id="tr-1", rationale="Looks fine")
    assert IssueEvidence.from_assessment(feedback) is None
