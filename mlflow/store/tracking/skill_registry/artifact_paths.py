"""
Artifact locations for skill content stored in MLflow artifact storage.

A ``source_type="mlflow"`` skill version records where its content lives in one of two ways,
and the difference decides who may delete the bytes:

- **Owned upload.** A standalone registration uploads content to a path the server generates,
  ``skills/[@<organization>/]<name>/<token>``, and records it with no ``subpath``. That path
  belongs to exactly one version and is reclaimed when the version is hard-deleted.
- **Referenced content.** A version can point into an existing tree with a ``subpath``.
  That content is not owned by the skill and is never reclaimed by deleting it.

Ownership is decided from the recorded fields alone, and only the exact shape the upload flow
writes counts as owned, so a row can never cause anything else to be deleted.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from mlflow.entities.skill_source import SkillSourceType
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import _decode
from mlflow.utils.validation import (
    _validate_organization_name,
    _validate_skill_name,
)

MLFLOW_ARTIFACTS_URI_PREFIX = "mlflow-artifacts:/"
SKILL_UPLOAD_ROOT = "skills"
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")


def is_skill_upload_namespace(artifact_path: str) -> bool:
    """Whether a proxy path can resolve into the server-owned Skill upload tree."""
    try:
        decoded = _decode(artifact_path)
    except ValueError:
        # The artifact handler rejects paths that cannot be fully decoded.
        return False
    segments = [segment for segment in decoded.split("/") if segment not in ("", ".")]
    if len(segments) >= 2 and segments[0].lower() == "workspaces":
        segments = segments[2:]
    return bool(segments and segments[0].lower() == SKILL_UPLOAD_ROOT)


def _identity_prefix(root: str, name: str, organization: str) -> str:
    if organization:
        return f"{root}/@{organization}/{name}"
    return f"{root}/{name}"


@dataclass(frozen=True)
class SkillArtifactIdentity:
    """The skill whose uploaded content an artifact path belongs to."""

    organization: str
    name: str


def parse_skill_upload_path(artifact_path: str) -> SkillArtifactIdentity | None:
    """
    The skill identity an artifact path under the upload root belongs to, or ``None``.

    Accepts a token path, anything beneath it, and the ancestor paths a listing can address
    (``skills``, ``skills/@<organization>``, ``skills/[@<organization>/]<name>``), so a
    permission check on any artifact request for skill content can resolve the skill to
    check. ``skills`` alone and ``skills/@<organization>`` alone name no single skill and
    return ``None``, as does anything outside the upload root or with a malformed identity.

    The path is resolved the way the artifact handlers will serve it: percent-escapes are
    decoded repeatedly, as ``validate_path_is_safe`` does, so an encoded segment cannot hide
    from the check, and any ``..``, ``.`` or empty segment fails closed instead of resolving
    past it to the identity the path started with.
    """
    if not isinstance(artifact_path, str):
        return None
    try:
        artifact_path = _decode(artifact_path)
    except ValueError:
        return None
    segments = artifact_path.strip("/").split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return None
    if segments[0] != SKILL_UPLOAD_ROOT:
        return None
    rest = segments[1:]
    organization = ""
    if rest and rest[0].startswith("@"):
        organization = rest[0][1:]
        rest = rest[1:]
        if not organization:
            return None
    if not rest or not rest[0]:
        return None
    name = rest[0]
    try:
        _validate_skill_name(name)
        _validate_organization_name(organization)
    except MlflowException:
        return None
    return SkillArtifactIdentity(organization=organization, name=name)


def new_skill_upload_path(name: str, organization: str = "") -> str:
    """
    A fresh artifact path for one upload of skill ``name``.

    The trailing token is random and generated per call, so no two uploads share a path: not
    concurrent registrations of the same skill, not a version number reused after a hard
    delete, and not the residue of an upload that failed before its row was committed.
    """
    _validate_skill_name(name)
    _validate_organization_name(organization)
    return f"{_identity_prefix(SKILL_UPLOAD_ROOT, name, organization)}/{uuid.uuid4().hex}"


def to_artifact_uri(artifact_path: str) -> str:
    return f"{MLFLOW_ARTIFACTS_URI_PREFIX}{artifact_path}"


def artifact_path_from_uri(source: str) -> str | None:
    """The path inside MLflow artifact storage named by ``source``, or ``None`` if it is not one."""
    if not isinstance(source, str) or not source.startswith(MLFLOW_ARTIFACTS_URI_PREFIX):
        return None
    return source.removeprefix(MLFLOW_ARTIFACTS_URI_PREFIX).strip("/") or None


def owned_skill_upload_path(
    *,
    name: str,
    organization: str,
    source_type: str | None,
    source: str | None,
    subpath: str | None,
) -> str | None:
    """
    The artifact path a skill version owns, or ``None`` when it owns nothing.

    Only the exact form written by the upload flow qualifies: an ``mlflow`` source with no
    ``subpath`` whose path is this identity's upload prefix followed by one token segment.
    Everything else (external pointers, references into an existing tree, another identity's
    prefix, a shorter or longer path) is not owned, so it is never scheduled for cleanup.
    """
    if source_type != SkillSourceType.MLFLOW.value or subpath is not None:
        return None
    if (path := artifact_path_from_uri(source)) is None:
        return None
    prefix = _identity_prefix(SKILL_UPLOAD_ROOT, name, organization)
    head, _, token = path.rpartition("/")
    if head != prefix or _TOKEN_PATTERN.fullmatch(token) is None:
        return None
    return path
