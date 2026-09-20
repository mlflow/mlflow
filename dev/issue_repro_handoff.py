"""Typed, fail-closed reports for issue reproduction triage."""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
READY = "ready"
REQUIRES_FURTHER_TRIAGE = "requires-further-triage"
MAX_REPORT_BYTES = 32 * 1024
MAX_EXCERPT_BYTES = 2 * 1024
MAX_TEXT_BYTES = 2 * 1024
MAX_LIMITATIONS = 8
MIN_CONFIDENCE = 0.8
SCRATCH_PATH = "scratch/reproduce.py"

ISSUE_KINDS = frozenset({"bug", "feature_request", "unknown"})
SURFACES = frozenset({"python_core", "unsupported", "unknown"})
FIDELITY_VERDICTS = frozenset({"faithful", "partial", "not_reproduced", "manufactured", "unknown"})
FAILURE_ORIGINS = frozenset({
    "reported_symptom",
    "trivial_assertion",
    "unrelated_exception",
    "unknown",
})
FIX_SCOPES = frozenset({"small", "broad", "unknown"})
OUTCOMES = frozenset({READY, REQUIRES_FURTHER_TRIAGE})

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<prefix>[\"']?(?:api[_-]?key|authorization|password|passwd|secret|token|"
    r"x[_-]?api[_-]?key|anthropic[_-]?api[_-]?key|github[_-]?token)[\"']?\s*[:=]\s*)"
    r"(?:(?:bearer\s+)?[\"'][^\"'\r\n]*[\"']|(?:bearer\s+)?[^\s,;\r\n]+)"
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)(?P<prefix>^(?:authorization|proxy-authorization|x-api-key)\s*:\s*).+$"
)
_TOKEN_RE = re.compile(
    r"(?i)\b(?:github_pat_[A-Za-z0-9_]+|gh[oprsu]_[A-Za-z0-9]+|"
    r"sk-ant-[A-Za-z0-9_-]+|AKIA[A-Z0-9]{16})\b"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)


class InvalidHandoff(ValueError):
    """Raised when model output or a persisted report violates the contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidHandoff(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InvalidHandoff(f"invalid {name} fields")
    return value


def _strip_controls(value: str) -> str:
    return "".join(
        character
        for character in value
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )


def redact(value: str) -> str:
    """Remove controls and common credentials before output is retained."""
    value = _strip_controls(value)
    value = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", value)
    value = _AUTH_HEADER_RE.sub(lambda match: f"{match.group('prefix')}[REDACTED]", value)
    value = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group('prefix')}[REDACTED]", value)
    return _TOKEN_RE.sub("[REDACTED]", value)


def _text(
    value: object,
    *,
    name: str,
    limit: int = MAX_TEXT_BYTES,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise InvalidHandoff(f"invalid {name}")
    value = redact(value)
    if (not value and not allow_empty) or len(value.encode("utf-8")) > limit:
        raise InvalidHandoff(f"invalid {name}")
    return value


def _enum(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise InvalidHandoff(f"invalid {name}")
    return value


def _confidence(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise InvalidHandoff(f"invalid {name}")
    return float(value)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise InvalidHandoff(f"invalid {name}")
    return value


def _binding(
    value: object,
    *,
    expected_repository: str,
    expected_issue_number: int,
    expected_event_sha: str,
    expected_checkout_sha: str,
) -> dict[str, Any]:
    binding = _object(value, {"repository", "issue_number", "event_sha", "checkout_sha"}, "binding")
    repository = binding["repository"]
    issue_number = binding["issue_number"]
    if (
        not isinstance(repository, str)
        or not _REPOSITORY_RE.fullmatch(repository)
        or repository != expected_repository
        or isinstance(issue_number, bool)
        or not isinstance(issue_number, int)
        or issue_number <= 0
        or issue_number != expected_issue_number
    ):
        raise InvalidHandoff("handoff issue binding mismatch")
    event_sha = _sha(binding["event_sha"], "event SHA")
    checkout_sha = _sha(binding["checkout_sha"], "checkout SHA")
    if event_sha != expected_event_sha:
        raise InvalidHandoff("handoff event SHA mismatch")
    if checkout_sha != expected_checkout_sha:
        raise InvalidHandoff("handoff checkout SHA mismatch")
    return dict(binding)


def _execution(value: object) -> dict[str, Any]:
    execution = _object(
        value,
        {
            "script_path",
            "executed_sha",
            "stdout_excerpt",
            "stderr_excerpt",
            "exit_status",
            "duration_seconds",
            "timed_out",
            "output_limited",
        },
        "execution",
    )
    script_path = _text(execution["script_path"], name="script path", limit=200)
    path = PurePosixPath(script_path)
    if (
        script_path != SCRATCH_PATH
        or path.is_absolute()
        or "\\" in script_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise InvalidHandoff("invalid script path")
    executed_sha = execution["executed_sha"]
    if executed_sha is not None:
        executed_sha = _sha(executed_sha, "executed SHA")
    exit_status = execution["exit_status"]
    if exit_status is not None and (
        isinstance(exit_status, bool)
        or not isinstance(exit_status, int)
        or not -255 <= exit_status <= 255
    ):
        raise InvalidHandoff("invalid exit status")
    duration = execution["duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 <= duration <= 60
    ):
        raise InvalidHandoff("invalid duration")
    for field in ("timed_out", "output_limited"):
        if not isinstance(execution[field], bool):
            raise InvalidHandoff(f"invalid {field}")
    return {
        "script_path": script_path,
        "executed_sha": executed_sha,
        "stdout_excerpt": _text(
            execution["stdout_excerpt"],
            name="stdout excerpt",
            limit=MAX_EXCERPT_BYTES,
            allow_empty=True,
        ),
        "stderr_excerpt": _text(
            execution["stderr_excerpt"],
            name="stderr excerpt",
            limit=MAX_EXCERPT_BYTES,
            allow_empty=True,
        ),
        "exit_status": exit_status,
        "duration_seconds": float(duration),
        "timed_out": execution["timed_out"],
        "output_limited": execution["output_limited"],
    }


def has_enhancement_label(labels: Iterable[object]) -> bool:
    """Return whether trusted issue metadata contains the exact enhancement label."""
    for label in labels:
        name = label.get("name") if isinstance(label, dict) else label
        if name == "enhancement":
            return True
    return False


def determine_outcome(report: dict[str, Any], *, issue_labels: Iterable[object] = ()) -> str:
    """Apply the trusted policy; every incomplete or uncertain case fails closed."""
    execution = report["execution"]
    fidelity = report["fidelity"]
    proposed_fix = report["proposed_fix"]
    binding = report["binding"]
    ready = (
        not has_enhancement_label(issue_labels)
        and report["issue_kind"] == "bug"
        and report["surface"] == "python_core"
        and binding["event_sha"] == binding["checkout_sha"]
        and execution["executed_sha"] == binding["checkout_sha"]
        and execution["exit_status"] is not None
        and not execution["timed_out"]
        and not execution["output_limited"]
        and fidelity["verdict"] == "faithful"
        and fidelity["confidence"] >= MIN_CONFIDENCE
        and fidelity["failure_origin"] == "reported_symptom"
        and proposed_fix["scope"] == "small"
        and bool(proposed_fix["summary"])
        and proposed_fix["confidence"] >= MIN_CONFIDENCE
        and not report["environment_limitations"]
        and not report["safety_uncertainty"]
    )
    return READY if ready else REQUIRES_FURTHER_TRIAGE


def validate_handoff(
    value: object,
    *,
    expected_repository: str,
    expected_issue_number: int,
    expected_event_sha: str,
    expected_checkout_sha: str,
    issue_labels: Iterable[object] = (),
    now: object | None = None,
) -> dict[str, Any]:
    """Validate and sanitize model output, then add the trusted proposed outcome."""
    del now  # Retained temporarily for callers of the superseded timestamped contract.
    handoff = _object(
        value,
        {
            "schema_version",
            "binding",
            "model_identifier",
            "issue_kind",
            "surface",
            "claimed_symptom",
            "execution",
            "fidelity",
            "proposed_fix",
            "environment_limitations",
            "safety_uncertainty",
        },
        "handoff",
    )
    if handoff["schema_version"] != SCHEMA_VERSION:
        raise InvalidHandoff("unsupported schema version")
    binding = _binding(
        handoff["binding"],
        expected_repository=expected_repository,
        expected_issue_number=expected_issue_number,
        expected_event_sha=expected_event_sha,
        expected_checkout_sha=expected_checkout_sha,
    )
    fidelity = _object(handoff["fidelity"], {"verdict", "confidence", "failure_origin"}, "fidelity")
    proposed_fix = _object(
        handoff["proposed_fix"], {"scope", "summary", "confidence"}, "proposed fix"
    )
    limitations = handoff["environment_limitations"]
    if not isinstance(limitations, list) or len(limitations) > MAX_LIMITATIONS:
        raise InvalidHandoff("invalid environment limitations")
    if not isinstance(handoff["safety_uncertainty"], bool):
        raise InvalidHandoff("invalid safety uncertainty")
    report = {
        "schema_version": SCHEMA_VERSION,
        "binding": binding,
        "model_identifier": _text(handoff["model_identifier"], name="model identifier", limit=200),
        "issue_kind": _enum(handoff["issue_kind"], ISSUE_KINDS, "issue kind"),
        "surface": _enum(handoff["surface"], SURFACES, "surface"),
        "claimed_symptom": _text(handoff["claimed_symptom"], name="claimed symptom"),
        "execution": _execution(handoff["execution"]),
        "fidelity": {
            "verdict": _enum(fidelity["verdict"], FIDELITY_VERDICTS, "fidelity verdict"),
            "confidence": _confidence(fidelity["confidence"], "fidelity confidence"),
            "failure_origin": _enum(fidelity["failure_origin"], FAILURE_ORIGINS, "failure origin"),
        },
        "proposed_fix": {
            "scope": _enum(proposed_fix["scope"], FIX_SCOPES, "fix scope"),
            "summary": _text(
                proposed_fix["summary"], name="proposed fix summary", allow_empty=True
            ),
            "confidence": _confidence(proposed_fix["confidence"], "fix confidence"),
        },
        "environment_limitations": [
            _text(item, name="environment limitation", limit=1_000) for item in limitations
        ],
        "safety_uncertainty": handoff["safety_uncertainty"],
    }
    report["proposed_outcome"] = determine_outcome(report, issue_labels=issue_labels)
    encoded_report = json.dumps(report, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded_report) > MAX_REPORT_BYTES:
        raise InvalidHandoff("report is oversized")
    return report


def load_handoff_json(
    payload: str | bytes,
    *,
    expected_repository: str,
    expected_issue_number: int,
    expected_event_sha: str,
    expected_checkout_sha: str,
    issue_labels: Iterable[object] = (),
    now: object | None = None,
) -> dict[str, Any]:
    """Decode bounded model JSON and return a sanitized finalized report."""
    encoded = payload.encode() if isinstance(payload, str) else payload
    if not isinstance(encoded, bytes) or len(encoded) > MAX_REPORT_BYTES:
        raise InvalidHandoff("handoff JSON is oversized")
    try:
        value = json.loads(encoded.decode(), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidHandoff("malformed handoff JSON") from error
    return validate_handoff(
        value,
        expected_repository=expected_repository,
        expected_issue_number=expected_issue_number,
        expected_event_sha=expected_event_sha,
        expected_checkout_sha=expected_checkout_sha,
        issue_labels=issue_labels,
        now=now,
    )


def _markdown(value: object) -> str:
    escaped = html.escape(str(value), quote=True)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", escaped)


def render_handoff_markdown(report: dict[str, Any]) -> str:
    """Render only the compact, sanitized report; never raw model context."""
    execution = report["execution"]
    fidelity = report["fidelity"]
    proposed_fix = report["proposed_fix"]
    lines = [
        "<!-- issue-repro-triage:v1 -->",
        "## Reproduction triage",
        "",
        f"**Proposed outcome:** `{report['proposed_outcome']}`",
        f"**Issue:** {_markdown(report['issue_kind'])} / {_markdown(report['surface'])}",
        f"**Checked commit:** `{report['binding']['checkout_sha']}`",
        f"**Symptom:** {_markdown(report['claimed_symptom'])}",
        f"**Fidelity:** {_markdown(fidelity['verdict'])} ({fidelity['confidence']:.2f})",
        f"**Fix scope:** {_markdown(proposed_fix['scope'])} ({proposed_fix['confidence']:.2f})",
        f"**Proposed fix:** {_markdown(proposed_fix['summary'] or 'None')}",
        "",
        "### Bounded execution evidence",
        "",
        f"- Exit status: {_markdown(execution['exit_status'])}",
        f"- Duration: {execution['duration_seconds']:.3f}s",
        f"- Timed out: {_markdown(execution['timed_out'])}",
        f"- Output limited: {_markdown(execution['output_limited'])}",
        f"- stdout: `{_markdown(execution['stdout_excerpt'] or '(empty)')}`",
        f"- stderr: `{_markdown(execution['stderr_excerpt'] or '(empty)')}`",
        "",
        "### Environment limitations",
        "",
        *([f"- {_markdown(item)}" for item in report["environment_limitations"]] or ["- None."]),
    ]
    rendered = "\n".join(lines)
    if len(rendered.encode()) > MAX_REPORT_BYTES:
        raise InvalidHandoff("rendered report is oversized")
    return rendered


render_report_markdown = render_handoff_markdown
