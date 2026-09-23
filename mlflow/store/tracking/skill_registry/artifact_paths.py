"""
Artifact locations for skill content stored in MLflow artifact storage.

A ``source_type="mlflow"`` skill version records where its content lives in one of two ways,
and the difference decides who may delete the bytes:

- **Owned upload.** A standalone registration uploads content to a path the server generates,
  ``skills/[@<organization>/]<name>/<token>``, and records it with no ``subpath``. That path
  belongs to exactly one version and is reclaimed when the version is hard-deleted.
- **Referenced package tree.** A member imported from an MLflow-stored agent plugin records the
  plugin version's tree, ``agent-plugins/[@<organization>/]<name>/<token>``, plus a ``subpath``.
  The skill version only points into that tree. The tree belongs to the plugin version, may be
  shared by several members, and is never reclaimed by deleting a skill.

Ownership is decided from the recorded fields alone, and only the exact shape the upload flow
writes counts as owned, so a row can never cause anything else to be deleted.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.paths import normalize_subpath
from mlflow.utils.validation import (
    _validate_agent_plugin_name,
    _validate_organization_name,
    _validate_skill_artifact_path,
    _validate_skill_name,
)

MLFLOW_ARTIFACTS_URI_PREFIX = "mlflow-artifacts:/"
SKILL_UPLOAD_ROOT = "skills"
AGENT_PLUGIN_UPLOAD_ROOT = "agent-plugins"
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")


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
    """
    if not isinstance(artifact_path, str):
        return None
    segments = artifact_path.strip("/").split("/")
    if not segments or segments[0] != SKILL_UPLOAD_ROOT:
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
    Everything else (external pointers, references into a package tree, another identity's
    prefix, a shorter or longer path) is not owned, so it is never scheduled for cleanup.
    """
    if source_type != "mlflow" or subpath is not None:
        return None
    if (path := artifact_path_from_uri(source)) is None:
        return None
    prefix = _identity_prefix(SKILL_UPLOAD_ROOT, name, organization)
    head, _, token = path.rpartition("/")
    if head != prefix or _TOKEN_PATTERN.fullmatch(token) is None:
        return None
    return path


def validate_referenced_mlflow_source(source: str | None, subpath: str | None) -> tuple[str, str]:
    """
    Validate the pointer an imported member records into an MLflow-stored agent plugin tree.

    ``source`` must name a plugin version's upload path,
    ``mlflow-artifacts:/agent-plugins/[@<organization>/]<name>/<token>``, and ``subpath`` must
    locate the skill inside it. Returns the canonical ``(source, subpath)`` to persist. The
    pointer is a reference only: it never satisfies ``owned_skill_upload_path``.
    """
    if (path := artifact_path_from_uri(source)) is None:
        raise MlflowException.invalid_parameter_value(
            f"A referenced MLflow source must be an '{MLFLOW_ARTIFACTS_URI_PREFIX}' URI, "
            f"got {source!r}."
        )
    _validate_skill_artifact_path(path)
    segments = path.split("/")
    if segments[0] != AGENT_PLUGIN_UPLOAD_ROOT:
        raise MlflowException.invalid_parameter_value(
            f"A referenced MLflow source must point into '{AGENT_PLUGIN_UPLOAD_ROOT}/', "
            f"got {source!r}."
        )
    match segments[1:]:
        case [organization, name, token] if organization.startswith("@"):
            _validate_organization_name(organization.removeprefix("@"))
            if not organization.removeprefix("@"):
                raise MlflowException.invalid_parameter_value(
                    f"Referenced MLflow source {source!r} has an empty organization segment."
                )
        case [name, token] if not name.startswith("@"):
            pass
        case _:
            raise MlflowException.invalid_parameter_value(
                f"Referenced MLflow source {source!r} must have the form "
                f"'{MLFLOW_ARTIFACTS_URI_PREFIX}{AGENT_PLUGIN_UPLOAD_ROOT}/"
                "[@<organization>/]<name>/<token>'."
            )
    _validate_agent_plugin_name(name)
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise MlflowException.invalid_parameter_value(
            f"Referenced MLflow source {source!r} does not end in a valid upload token."
        )
    if (normalized_subpath := normalize_subpath(subpath)) is None:
        raise MlflowException.invalid_parameter_value(
            "A referenced MLflow source requires a 'subpath' locating the skill inside the "
            "package tree."
        )
    return to_artifact_uri(path), normalized_subpath
