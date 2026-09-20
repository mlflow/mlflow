import copy

import pytest

from dev.issue_repro_policy import READY, REQUIRES_FURTHER_TRIAGE, determine_outcome

EXPECTED_BINDING = {
    "repository": "mlflow/mlflow",
    "issue_number": 42,
    "event_sha": "a" * 40,
    "checkout_sha": "b" * 40,
}


def _inputs():
    artifact = {
        "relative_path": "scratch/reproduce.py",
        "sha256": "c" * 64,
        "size_bytes": 24,
    }
    return {
        "validated_handoff": {
            "binding": copy.deepcopy(EXPECTED_BINDING),
            "issue_kind": "bug",
            "symptoms": [
                {
                    "claim": "The validator raises ValueError.",
                    "verdict": "reproduced",
                    "reproduction_steps": ["Run the scratch reproduction."],
                    "stdout": "",
                    "stderr": "ValueError: invalid model URI\n",
                    "exit_status": 1,
                    "duration_seconds": 0.2,
                    "artifact": artifact,
                }
            ],
            "overall_verdict": "reproduced",
        },
        "verifier_judgments": [
            {
                "symptom_index": 0,
                "verdict": "reproduced",
                "confidence": 0.8,
                "reason": "The raw exception matches the reported symptom.",
                "faithful_reproduction": True,
            }
        ],
        "fix_scope_judgment": {
            "issue_kind": "bug",
            "fix_scope": "small",
            "proposed_fix_present": True,
            "confidence": 0.8,
            "reason": "The explicit proposal is localized.",
        },
        "issue_labels": ["bug"],
        "expected_binding": copy.deepcopy(EXPECTED_BINDING),
        "safety_checks": {
            "broker_safety_passed": True,
            "current_master_reproduction": True,
            "historical_environment_required": False,
            "supported_surface": "python_core",
        },
    }


def test_complete_positive_case_is_ready():
    assert determine_outcome(**_inputs()) == READY


@pytest.mark.parametrize(
    ("description", "mutate"),
    [
        ("missing handoff", lambda value: value.update(validated_handoff=None)),
        (
            "feature request",
            lambda value: value["validated_handoff"].update(issue_kind="feature_request"),
        ),
        (
            "unknown issue kind",
            lambda value: value["validated_handoff"].update(issue_kind="unknown"),
        ),
        ("enhancement label", lambda value: value.update(issue_labels=["bug", "enhancement"])),
        ("malformed labels", lambda value: value.update(issue_labels=[{"color": "red"}])),
        (
            "cross-issue binding",
            lambda value: value["validated_handoff"]["binding"].update(issue_number=43),
        ),
        (
            "cross-SHA binding",
            lambda value: value["validated_handoff"]["binding"].update(checkout_sha="d" * 40),
        ),
        (
            "malformed expected binding",
            lambda value: value["expected_binding"].update(extra=True),
        ),
        (
            "unsupported surface",
            lambda value: value["safety_checks"].update(supported_surface="javascript_ui"),
        ),
        (
            "historical environment required",
            lambda value: value["safety_checks"].update(historical_environment_required=True),
        ),
        (
            "not current master",
            lambda value: value["safety_checks"].update(current_master_reproduction=False),
        ),
        (
            "broker safety failure",
            lambda value: value["safety_checks"].update(broker_safety_passed=False),
        ),
        (
            "unknown safety check",
            lambda value: value["safety_checks"].update(secret_scan_passed=True),
        ),
        (
            "not reproduced overall",
            lambda value: value["validated_handoff"].update(overall_verdict="not_reproduced"),
        ),
        (
            "manual review overall",
            lambda value: value["validated_handoff"].update(overall_verdict="needs_manual_review"),
        ),
        (
            "inconclusive handoff symptom",
            lambda value: value["validated_handoff"]["symptoms"][0].update(verdict="inconclusive"),
        ),
        (
            "missing raw output",
            lambda value: value["validated_handoff"]["symptoms"][0].update(stdout="", stderr=""),
        ),
        (
            "missing reproduction artifact",
            lambda value: value["validated_handoff"]["symptoms"][0].update(artifact=None),
        ),
        ("missing verifier result", lambda value: value.update(verifier_judgments=[])),
        (
            "inconclusive verifier",
            lambda value: value["verifier_judgments"][0].update(
                verdict="inconclusive", faithful_reproduction=False
            ),
        ),
        (
            "contradictory verifier",
            lambda value: value["verifier_judgments"][0].update(faithful_reproduction=False),
        ),
        (
            "low verifier confidence",
            lambda value: value["verifier_judgments"][0].update(confidence=0.79),
        ),
        (
            "malformed verifier confidence",
            lambda value: value["verifier_judgments"][0].update(confidence=True),
        ),
        (
            "duplicate verifier result",
            lambda value: value["verifier_judgments"].append(
                copy.deepcopy(value["verifier_judgments"][0])
            ),
        ),
        (
            "feature request fix judgment",
            lambda value: value["fix_scope_judgment"].update(issue_kind="feature_request"),
        ),
        (
            "broad fix",
            lambda value: value["fix_scope_judgment"].update(fix_scope="broad"),
        ),
        (
            "unknown fix",
            lambda value: value["fix_scope_judgment"].update(fix_scope="unknown"),
        ),
        (
            "no explicit proposed fix",
            lambda value: value["fix_scope_judgment"].update(proposed_fix_present=False),
        ),
        (
            "low fix confidence",
            lambda value: value["fix_scope_judgment"].update(confidence=0.79),
        ),
        (
            "model-selected label injection",
            lambda value: value["fix_scope_judgment"].update(label="ready"),
        ),
        ("agent failure", lambda value: value.update(verifier_judgments=None)),
        ("model failure", lambda value: value.update(fix_scope_judgment=None)),
    ],
)
def test_every_incomplete_risky_or_uncertain_case_fails_closed(description, mutate):
    inputs = _inputs()
    mutate(inputs)

    assert determine_outcome(**inputs) == REQUIRES_FURTHER_TRIAGE, description


def test_every_claim_must_pass_independent_verification():
    inputs = _inputs()
    second_symptom = copy.deepcopy(inputs["validated_handoff"]["symptoms"][0])
    second_symptom["claim"] = "The command also prints a warning."
    inputs["validated_handoff"]["symptoms"].append(second_symptom)
    second_verification = copy.deepcopy(inputs["verifier_judgments"][0])
    second_verification.update(symptom_index=1, verdict="inconclusive", faithful_reproduction=False)
    inputs["verifier_judgments"].append(second_verification)

    assert determine_outcome(**inputs) == REQUIRES_FURTHER_TRIAGE
