"""Validation and rendering for issue reproduction handoffs."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
MAX_HANDOFF_AGE = timedelta(hours=6)
MAX_HANDOFF_JSON = 100_000
MAX_CLOCK_SKEW = timedelta(minutes=5)
MAX_SYMPTOMS = 20
MAX_STEPS = 20
MAX_CLAIM = 2_000
MAX_STEP = 1_000
MAX_OUTPUT = 6_000
MAX_LIMITATIONS = 20
MAX_LIMITATION = 1_000

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PYTHON_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?")
_ARTIFACT_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ISSUE_KINDS = {"bug", "feature_request", "unknown"}
_SYMPTOM_VERDICTS = {"reproduced", "not_reproduced", "inconclusive"}
_OVERALL_VERDICTS = {"reproduced", "not_reproduced", "needs_manual_review"}


class InvalidHandoff(ValueError):
    """Raised when a reproduction handoff violates its trusted contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InvalidHandoff(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InvalidHandoff(f"invalid {name} fields")
    return value


def _string(value: object, *, name: str, limit: int, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > limit
        or (not allow_empty and not value)
        or any(ord(character) < 32 and character not in "\n\t" for character in value)
    ):
        raise InvalidHandoff(f"invalid {name}")
    return value


def _timestamp(value: object, name: str) -> datetime:
    value = _string(value, name=name, limit=40)
    if not value.endswith("Z"):
        raise InvalidHandoff(f"invalid {name}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise InvalidHandoff(f"invalid {name}") from error
    if parsed.tzinfo != timezone.utc:
        raise InvalidHandoff(f"invalid {name}")
    return parsed


def _relative_artifact_path(value: object) -> str:
    value = _string(value, name="artifact path", limit=500)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith("scratch/")
    ):
        raise InvalidHandoff("invalid artifact path")
    return value


def _artifact(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    artifact = _object(
        value,
        {"relative_path", "sha256", "size_bytes"},
        "artifact",
    )
    _relative_artifact_path(artifact["relative_path"])
    if not isinstance(artifact["sha256"], str) or not _ARTIFACT_SHA_RE.fullmatch(
        artifact["sha256"]
    ):
        raise InvalidHandoff("invalid artifact sha256")
    if (
        isinstance(artifact["size_bytes"], bool)
        or not isinstance(artifact["size_bytes"], int)
        or not 0 <= artifact["size_bytes"] <= 100_000
    ):
        raise InvalidHandoff("invalid artifact size")
    return artifact


def _symptom(value: object, index: int) -> dict[str, Any]:
    symptom = _object(
        value,
        {
            "claim",
            "verdict",
            "reproduction_steps",
            "stdout",
            "stderr",
            "exit_status",
            "duration_seconds",
            "artifact",
        },
        f"symptom {index}",
    )
    _string(symptom["claim"], name="symptom claim", limit=MAX_CLAIM)
    if symptom["verdict"] not in _SYMPTOM_VERDICTS:
        raise InvalidHandoff("invalid symptom verdict")
    steps = symptom["reproduction_steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise InvalidHandoff("invalid reproduction steps")
    for step in steps:
        _string(step, name="reproduction step", limit=MAX_STEP)
    stdout = _string(symptom["stdout"], name="stdout", limit=MAX_OUTPUT, allow_empty=True)
    stderr = _string(symptom["stderr"], name="stderr", limit=MAX_OUTPUT, allow_empty=True)
    status = symptom["exit_status"]
    if status is not None and (
        isinstance(status, bool) or not isinstance(status, int) or not -255 <= status <= 255
    ):
        raise InvalidHandoff("invalid exit status")
    duration = symptom["duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 <= duration <= 600
    ):
        raise InvalidHandoff("invalid duration")
    if not stdout and not stderr and status is None:
        raise InvalidHandoff("symptom is missing raw evidence")
    _artifact(symptom["artifact"])
    return symptom


def validate_handoff(
    value: object,
    *,
    expected_repository: str,
    expected_issue_number: int,
    expected_event_sha: str,
    expected_checkout_sha: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a handoff and its bindings, returning the original typed value."""
    handoff = _object(
        value,
        {
            "schema_version",
            "binding",
            "runner",
            "started_at",
            "completed_at",
            "issue_kind",
            "symptoms",
            "overall_verdict",
            "environment_limitations",
            "confidence",
        },
        "handoff",
    )
    if handoff["schema_version"] != SCHEMA_VERSION:
        raise InvalidHandoff("unsupported schema version")

    binding = _object(
        handoff["binding"],
        {"repository", "issue_number", "event_sha", "checkout_sha"},
        "binding",
    )
    if (
        not isinstance(binding["repository"], str)
        or not _REPOSITORY_RE.fullmatch(binding["repository"])
        or binding["repository"] != expected_repository
        or isinstance(binding["issue_number"], bool)
        or not isinstance(binding["issue_number"], int)
        or binding["issue_number"] <= 0
        or binding["issue_number"] != expected_issue_number
    ):
        raise InvalidHandoff("handoff issue binding mismatch")
    for field, expected in (
        ("event_sha", expected_event_sha),
        ("checkout_sha", expected_checkout_sha),
    ):
        if not isinstance(binding[field], str) or not _SHA_RE.fullmatch(binding[field]):
            raise InvalidHandoff(f"invalid {field}")
        if binding[field] != expected:
            raise InvalidHandoff(f"handoff {field} mismatch")

    runner = _object(handoff["runner"], {"os", "architecture", "python_version"}, "runner")
    _string(runner["os"], name="runner os", limit=100)
    _string(runner["architecture"], name="runner architecture", limit=100)
    if not isinstance(runner["python_version"], str) or not _PYTHON_VERSION_RE.fullmatch(
        runner["python_version"]
    ):
        raise InvalidHandoff("invalid Python version")

    started = _timestamp(handoff["started_at"], "started_at")
    completed = _timestamp(handoff["completed_at"], "completed_at")
    current_time = now or datetime.now(timezone.utc)
    if started > completed or completed < current_time - MAX_HANDOFF_AGE:
        raise InvalidHandoff("stale handoff timestamps")
    if completed > current_time + MAX_CLOCK_SKEW:
        raise InvalidHandoff("future handoff timestamp")

    if handoff["issue_kind"] not in _ISSUE_KINDS:
        raise InvalidHandoff("invalid issue kind")
    symptoms = handoff["symptoms"]
    if not isinstance(symptoms, list) or not 1 <= len(symptoms) <= MAX_SYMPTOMS:
        raise InvalidHandoff("invalid symptoms")
    validated_symptoms = [_symptom(symptom, index) for index, symptom in enumerate(symptoms)]

    overall = handoff["overall_verdict"]
    if overall not in _OVERALL_VERDICTS:
        raise InvalidHandoff("invalid overall verdict")
    symptom_verdicts = {symptom["verdict"] for symptom in validated_symptoms}
    if overall == "reproduced" and symptom_verdicts != {"reproduced"}:
        raise InvalidHandoff("overall verdict contradicts symptom verdicts")
    if overall == "not_reproduced" and symptom_verdicts != {"not_reproduced"}:
        raise InvalidHandoff("overall verdict contradicts symptom verdicts")
    if overall == "needs_manual_review" and symptom_verdicts in (
        {"reproduced"},
        {"not_reproduced"},
    ):
        raise InvalidHandoff("overall verdict contradicts symptom verdicts")

    limitations = handoff["environment_limitations"]
    if not isinstance(limitations, list) or len(limitations) > MAX_LIMITATIONS:
        raise InvalidHandoff("invalid environment limitations")
    for limitation in limitations:
        _string(limitation, name="environment limitation", limit=MAX_LIMITATION)
    confidence = handoff["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise InvalidHandoff("invalid confidence")
    return handoff


def load_handoff_json(
    payload: str | bytes,
    *,
    expected_repository: str,
    expected_issue_number: int,
    expected_event_sha: str,
    expected_checkout_sha: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decode bounded JSON and validate it as a bound reproduction handoff."""
    try:
        encoded = payload.encode() if isinstance(payload, str) else payload
        if not isinstance(encoded, bytes) or len(encoded) > MAX_HANDOFF_JSON:
            raise InvalidHandoff("handoff JSON is oversized")
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidHandoff("malformed handoff JSON") from error
    return validate_handoff(
        value,
        expected_repository=expected_repository,
        expected_issue_number=expected_issue_number,
        expected_event_sha=expected_event_sha,
        expected_checkout_sha=expected_checkout_sha,
        now=now,
    )


def _plain(value: object) -> str:
    markdown_punctuation = "\\`*_{}[]()#+.!|"
    return "".join(
        f"&#{ord(character)};"
        if character in markdown_punctuation
        else html.escape(character, quote=True)
        for character in str(value)
    )


def _evidence(value: str) -> str:
    return html.escape(value or "(empty)", quote=True).replace("`", "&#96;")


def render_handoff_markdown(handoff: dict[str, Any]) -> str:
    """Render only validated handoff fields into deterministic, escaped Markdown."""
    binding = handoff["binding"]
    runner = handoff["runner"]
    lines = [
        "<!-- issue-repro-triage:v1 -->",
        "## Reproduction handoff",
        "",
        f"**Overall verdict:** {_plain(handoff['overall_verdict'].replace('_', ' '))}",
        f"**Issue kind:** {_plain(handoff['issue_kind'].replace('_', ' '))}",
        f"**Confidence:** {handoff['confidence']:.2f}",
        f"**Source:** `{binding['checkout_sha']}`",
        f"**Environment:** {_plain(runner['os'])} / {_plain(runner['architecture'])} / "
        f"Python {_plain(runner['python_version'])}",
        "",
        "### Symptoms",
    ]
    for index, symptom in enumerate(handoff["symptoms"], start=1):
        lines.extend([
            "",
            f"#### {index}. {_plain(symptom['claim'])}",
            "",
            f"**Verdict:** {_plain(symptom['verdict'].replace('_', ' '))}",
            f"**Exit status:** {_plain(symptom['exit_status'])}",
            f"**Duration:** {symptom['duration_seconds']:.3f}s",
            "**Steps:**",
            "",
            *[
                f"{step_index}. {_plain(step)}"
                for step_index, step in enumerate(symptom["reproduction_steps"], 1)
            ],
            "",
            "<details>",
            "<summary>Raw stdout and stderr</summary>",
            "",
            "**stdout**",
            f"<pre>{_evidence(symptom['stdout'])}</pre>",
            "",
            "**stderr**",
            f"<pre>{_evidence(symptom['stderr'])}</pre>",
            "",
            "</details>",
        ])
    lines.extend(["", "### Environment limitations", ""])
    if handoff["environment_limitations"]:
        lines.extend(f"- {_plain(item)}" for item in handoff["environment_limitations"])
    else:
        lines.append("- None recorded.")
    return "\n".join(lines)


def artifact_sha256(content: bytes) -> str:
    """Return the digest recorded for a bounded scratch reproduction artifact."""
    return hashlib.sha256(content).hexdigest()
