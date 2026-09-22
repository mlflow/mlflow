import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from issue_repro_handoff import (
    MAX_EXCERPT_BYTES,
    InvalidHandoff,
    has_enhancement_label,
    load_handoff_json,
    render_handoff_markdown,
    validate_handoff,
)

SHA = "a" * 40


def _handoff():
    return {
        "schema_version": 1,
        "binding": {
            "repository": "mlflow/mlflow",
            "issue_number": 42,
            "event_sha": SHA,
            "checkout_sha": SHA,
        },
        "model_identifier": "claude-sonnet-4-6",
        "issue_kind": "bug",
        "surface": "python_core",
        "claimed_symptom": "search_traces returns duplicate rows",
        "execution": {
            "script_path": "scratch/reproduce.py",
            "executed_sha": SHA,
            "stdout_excerpt": "expected 2 rows, got 4",
            "stderr_excerpt": "",
            "exit_status": 1,
            "duration_seconds": 1.25,
            "timed_out": False,
            "output_limited": False,
        },
        "fidelity": {
            "verdict": "faithful",
            "confidence": 0.8,
            "failure_origin": "reported_symptom",
        },
        "proposed_fix": {
            "scope": "small",
            "summary": "Apply the existing row limit before materializing results.",
            "confidence": 0.8,
        },
        "environment_limitations": [],
        "safety_uncertainty": False,
    }


def _validate(value, **kwargs):
    return validate_handoff(
        value,
        expected_repository="mlflow/mlflow",
        expected_issue_number=42,
        expected_event_sha=SHA,
        expected_checkout_sha=SHA,
        **kwargs,
    )


def test_only_complete_high_confidence_case_is_ready():
    report = _validate(_handoff())

    assert report["proposed_outcome"] == "ready"
    assert "proposed_outcome" not in _handoff()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(issue_kind="feature_request"),
        lambda value: value.update(issue_kind="unknown"),
        lambda value: value.update(surface="unsupported"),
        lambda value: value.update(surface="unknown"),
        lambda value: value["execution"].update(executed_sha="b" * 40),
        lambda value: value["execution"].update(executed_sha=None),
        lambda value: value["execution"].update(exit_status=None),
        lambda value: value["execution"].update(timed_out=True),
        lambda value: value["execution"].update(output_limited=True),
        lambda value: value["fidelity"].update(verdict="partial"),
        lambda value: value["fidelity"].update(verdict="not_reproduced"),
        lambda value: value["fidelity"].update(verdict="manufactured"),
        lambda value: value["fidelity"].update(confidence=0.79),
        lambda value: value["fidelity"].update(failure_origin="trivial_assertion"),
        lambda value: value["fidelity"].update(failure_origin="unrelated_exception"),
        lambda value: value["proposed_fix"].update(scope="broad"),
        lambda value: value["proposed_fix"].update(scope="unknown"),
        lambda value: value["proposed_fix"].update(summary=""),
        lambda value: value["proposed_fix"].update(confidence=0.79),
        lambda value: value.update(environment_limitations=["CUDA was unavailable."]),
        lambda value: value.update(safety_uncertainty=True),
    ],
)
def test_incomplete_or_uncertain_cases_require_further_triage(mutate):
    handoff = _handoff()
    mutate(handoff)

    assert _validate(handoff)["proposed_outcome"] == "requires-further-triage"


def test_enhancement_label_short_circuits_even_complete_bug_output():
    assert has_enhancement_label([{"name": "enhancement"}])
    assert not has_enhancement_label(["Enhancement"])
    assert (
        _validate(_handoff(), issue_labels=[{"name": "enhancement"}])["proposed_outcome"]
        == "requires-further-triage"
    )
    assert _validate(_handoff(), issue_labels=["Enhancement"])["proposed_outcome"] == "ready"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "handoff fields"),
        (lambda value: value.update(proposed_outcome="ready"), "handoff fields"),
        (lambda value: value["execution"].update(extra=True), "execution fields"),
        (lambda value: value.update(issue_kind="incident"), "issue kind"),
        (lambda value: value.update(surface="ui"), "surface"),
        (lambda value: value["fidelity"].update(verdict="yes"), "fidelity verdict"),
        (lambda value: value["proposed_fix"].update(scope="medium"), "fix scope"),
        (lambda value: value["fidelity"].update(confidence=float("nan")), "confidence"),
        (lambda value: value["execution"].update(duration_seconds=61), "duration"),
        (lambda value: value["execution"].update(timed_out="false"), "timed_out"),
    ],
)
def test_rejects_extra_fields_invalid_enums_and_invalid_types(mutate, message):
    handoff = _handoff()
    mutate(handoff)

    with pytest.raises(InvalidHandoff, match=message):
        _validate(handoff)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "attacker/repo", "issue binding"),
        ("issue_number", 43, "issue binding"),
        ("event_sha", "b" * 40, "event SHA mismatch"),
        ("checkout_sha", "b" * 40, "checkout SHA mismatch"),
    ],
)
def test_rejects_cross_issue_and_cross_sha_bindings(field, value, message):
    handoff = _handoff()
    handoff["binding"][field] = value

    with pytest.raises(InvalidHandoff, match=message):
        _validate(handoff)


def test_stale_execution_sha_fails_closed():
    handoff = _handoff()
    handoff["execution"]["executed_sha"] = "b" * 40

    assert _validate(handoff)["proposed_outcome"] == "requires-further-triage"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/reproduce.py",
        "scratch/../secrets",
        "../scratch/reproduce.py",
        "scratch\\reproduce.py",
        "tests/reproduce.py",
    ],
)
def test_rejects_traversal_and_alternate_script_paths(path):
    handoff = _handoff()
    handoff["execution"]["script_path"] = path

    with pytest.raises(InvalidHandoff, match="script path"):
        _validate(handoff)


def test_rejects_oversized_excerpt_and_payload():
    handoff = _handoff()
    handoff["execution"]["stdout_excerpt"] = "x" * (MAX_EXCERPT_BYTES + 1)
    with pytest.raises(InvalidHandoff, match="stdout excerpt"):
        _validate(handoff)

    with pytest.raises(InvalidHandoff, match="oversized"):
        load_handoff_json(
            " " * (32 * 1024 + 1),
            expected_repository="mlflow/mlflow",
            expected_issue_number=42,
            expected_event_sha=SHA,
            expected_checkout_sha=SHA,
        )


def test_load_rejects_duplicate_fields():
    payload = json.dumps(_handoff()).replace(
        '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
    )

    with pytest.raises(InvalidHandoff, match="duplicate JSON field"):
        load_handoff_json(
            payload,
            expected_repository="mlflow/mlflow",
            expected_issue_number=42,
            expected_event_sha=SHA,
            expected_checkout_sha=SHA,
        )


def test_strips_controls_and_redacts_secrets_before_persistence_and_rendering():
    handoff = _handoff()
    handoff["claimed_symptom"] = "bad\x00 rows; token=plain-secret"
    handoff["execution"].update(
        stdout_excerpt=(
            "Authorization: Bearer auth-secret\n"
            "x-api-key: key-secret\n"
            "github_pat_abcdefghijklmnopqrstuvwxyz\n"
            "sk-ant-api03-secretvalue"
        ),
        stderr_excerpt='password="hunter2"; ANTHROPIC_API_KEY=secret-key',
    )

    report = _validate(handoff)
    serialized = json.dumps(report)
    rendered = render_handoff_markdown(report)

    for secret in (
        "plain-secret",
        "auth-secret",
        "key-secret",
        "github_pat_abcdefghijklmnopqrstuvwxyz",
        "sk-ant-api03-secretvalue",
        "hunter2",
        "secret-key",
    ):
        assert secret not in serialized
        assert secret not in rendered
    assert "\x00" not in serialized
    assert serialized.count("[REDACTED]") >= 6


def test_rendering_escapes_markdown_and_html_injection_and_is_bounded():
    handoff = _handoff()
    handoff["claimed_symptom"] = "<script>alert(1)</script> [click](javascript:alert(1))"
    handoff["proposed_fix"]["summary"] = "Use `dangerous()` **now**"
    report = _validate(handoff)

    first = render_handoff_markdown(report)
    second = render_handoff_markdown(copy.deepcopy(report))

    assert first == second
    assert len(first.encode()) <= 32 * 1024
    assert "<script>" not in first
    assert "[click](javascript:" not in first
    assert "&lt;script&gt;" in first
    assert "\\[click\\]\\(javascript:alert\\(1\\)\\)" in first
