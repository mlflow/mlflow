"""Fail-closed outcome policy for issue reproduction triage."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Any

READY = "ready"
REQUIRES_FURTHER_TRIAGE = "requires-further-triage"
MIN_CONFIDENCE = 0.8

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_SHA_RE = re.compile(r"[0-9a-f]{64}")
_BINDING_FIELDS = {"repository", "issue_number", "event_sha", "checkout_sha"}
_SAFETY_FIELDS = {
    "broker_safety_passed",
    "current_master_reproduction",
    "historical_environment_required",
    "supported_surface",
}
_SYMPTOM_FIELDS = {
    "claim",
    "verdict",
    "reproduction_steps",
    "stdout",
    "stderr",
    "exit_status",
    "duration_seconds",
    "artifact",
}
_VERIFICATION_FIELDS = {
    "symptom_index",
    "verdict",
    "confidence",
    "reason",
    "faithful_reproduction",
}
_FIX_SCOPE_FIELDS = {
    "issue_kind",
    "fix_scope",
    "proposed_fix_present",
    "confidence",
    "reason",
}


def _confidence_at_least(value: object, minimum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and minimum <= value <= 1
    )


def _binding_matches(binding: object, expected: object) -> bool:
    if (
        not isinstance(binding, dict)
        or not isinstance(expected, dict)
        or set(binding) != _BINDING_FIELDS
        or set(expected) != _BINDING_FIELDS
        or binding != expected
    ):
        return False
    return (
        isinstance(binding["repository"], str)
        and "/" in binding["repository"]
        and not isinstance(binding["issue_number"], bool)
        and isinstance(binding["issue_number"], int)
        and binding["issue_number"] > 0
        and isinstance(binding["event_sha"], str)
        and _SHA_RE.fullmatch(binding["event_sha"]) is not None
        and isinstance(binding["checkout_sha"], str)
        and _SHA_RE.fullmatch(binding["checkout_sha"]) is not None
    )


def _labels_have_enhancement(labels: object) -> bool | None:
    if not isinstance(labels, list):
        return None
    names = []
    for label in labels:
        name = label.get("name") if isinstance(label, dict) else label
        if not isinstance(name, str) or not name:
            return None
        names.append(name.casefold())
    return "enhancement" in names


def _safety_passes(checks: object) -> bool:
    return (
        isinstance(checks, dict)
        and set(checks) == _SAFETY_FIELDS
        and checks["broker_safety_passed"] is True
        and checks["current_master_reproduction"] is True
        and checks["historical_environment_required"] is False
        and checks["supported_surface"] == "python_core"
    )


def _symptoms_pass(handoff: dict[str, Any], verifications: object) -> bool:
    symptoms = handoff.get("symptoms")
    if (
        handoff.get("overall_verdict") != "reproduced"
        or not isinstance(symptoms, list)
        or not symptoms
        or not isinstance(verifications, list)
        or len(verifications) != len(symptoms)
    ):
        return False

    verification_by_index: dict[int, dict[str, Any]] = {}
    for verification in verifications:
        if not isinstance(verification, dict) or set(verification) != _VERIFICATION_FIELDS:
            return False
        index = verification["symptom_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(symptoms)
            or index in verification_by_index
        ):
            return False
        verification_by_index[index] = verification

    for index, symptom in enumerate(symptoms):
        if not isinstance(symptom, dict) or set(symptom) != _SYMPTOM_FIELDS:
            return False
        artifact = symptom["artifact"]
        artifact_path = artifact.get("relative_path") if isinstance(artifact, dict) else None
        path = PurePosixPath(artifact_path) if isinstance(artifact_path, str) else None
        has_artifact = (
            isinstance(artifact, dict)
            and set(artifact) == {"relative_path", "sha256", "size_bytes"}
            and path is not None
            and not path.is_absolute()
            and path.parts[:1] == ("scratch",)
            and ".." not in path.parts
            and isinstance(artifact["sha256"], str)
            and _ARTIFACT_SHA_RE.fullmatch(artifact["sha256"]) is not None
            and not isinstance(artifact["size_bytes"], bool)
            and isinstance(artifact["size_bytes"], int)
            and artifact["size_bytes"] > 0
        )
        has_raw_output = (
            isinstance(symptom["stdout"], str)
            and isinstance(symptom["stderr"], str)
            and bool(symptom["stdout"] or symptom["stderr"])
        )
        verification = verification_by_index.get(index)
        if (
            not isinstance(symptom["claim"], str)
            or not symptom["claim"]
            or symptom["verdict"] != "reproduced"
            or not has_raw_output
            or not has_artifact
            or verification is None
            or verification["verdict"] != "reproduced"
            or verification["faithful_reproduction"] is not True
            or not _confidence_at_least(verification["confidence"], MIN_CONFIDENCE)
            or not isinstance(verification["reason"], str)
            or not verification["reason"]
        ):
            return False
    return True


def _fix_scope_passes(fix_scope: object) -> bool:
    return (
        isinstance(fix_scope, dict)
        and set(fix_scope) == _FIX_SCOPE_FIELDS
        and fix_scope["issue_kind"] == "bug"
        and fix_scope["fix_scope"] == "small"
        and fix_scope["proposed_fix_present"] is True
        and _confidence_at_least(fix_scope["confidence"], MIN_CONFIDENCE)
        and isinstance(fix_scope["reason"], str)
        and bool(fix_scope["reason"])
    )


def determine_outcome(
    *,
    validated_handoff: object,
    verifier_judgments: object,
    fix_scope_judgment: object,
    issue_labels: object,
    expected_binding: object,
    safety_checks: object,
) -> str:
    """Compute the sole publishable outcome without trusting model-selected labels."""
    if not isinstance(validated_handoff, dict):
        return REQUIRES_FURTHER_TRIAGE
    has_enhancement = _labels_have_enhancement(issue_labels)
    if (
        has_enhancement is not False
        or validated_handoff.get("issue_kind") != "bug"
        or not _binding_matches(validated_handoff.get("binding"), expected_binding)
        or not _safety_passes(safety_checks)
        or not _symptoms_pass(validated_handoff, verifier_judgments)
        or not _fix_scope_passes(fix_scope_judgment)
    ):
        return REQUIRES_FURTHER_TRIAGE
    return READY
