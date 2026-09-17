from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import extract_skill_archive, get_max_decompressed_size
from mlflow.genai.skill_content.paths import collect_tree
from mlflow.genai.skill_content.sources import OCI_SCHEME, is_local_path, resolve_source_type
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_CONFLICT,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.server.skill_registry.artifacts import (
    delete_artifact_tree_best_effort,
    require_artifact_serving,
    store_skill_tree,
)
from mlflow.store.tracking.skill_registry.artifact_paths import (
    new_skill_upload_path,
    to_artifact_uri,
)
from mlflow.utils.validation import _validate_organization_name, _validate_skill_name

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_UPLOAD_CHUNK_SIZE = 1024 * 1024
# Gzip framing and tar headers can make an archive of incompressible content slightly larger
# than the content itself.
_UPLOAD_SIZE_SLACK = 1024 * 1024
_REGISTRATION_STATUSES = (SkillStatus.ACTIVE.value, SkillStatus.DRAFT.value)
_CLIENT_SOURCE_TYPES = (
    SkillSourceType.GIT.value,
    SkillSourceType.OCI.value,
    SkillSourceType.ZIP.value,
)
# Errors the store raises only when nothing was committed. After any other failure the row may
# exist, so content stored for it must be left alone.
_DEFINITE_REJECTIONS = frozenset(
    ErrorCode.Name(code)
    for code in (
        INVALID_PARAMETER_VALUE,
        RESOURCE_ALREADY_EXISTS,
        RESOURCE_CONFLICT,
        RESOURCE_DOES_NOT_EXIST,
    )
)


@dataclass(frozen=True)
class SkillVersionRegistration:
    """The metadata of a registration request. Content, when present, travels beside it."""

    name: str | None
    organization: str = ""
    source: str | None = None
    source_type: str | None = None
    ref: str | None = None
    subpath: str | None = None
    digest: str | None = None
    status: str = SkillStatus.ACTIVE.value


def register_skill_version(
    registration: SkillVersionRegistration,
    *,
    content: BinaryIO | None = None,
    multipart: bool = False,
) -> SkillVersion:
    """
    Register a skill version from a request that carries metadata and, for a local skill, the
    packaged content.

    The body and ``source`` must agree, and a mismatch is rejected rather than guessed at:

    - No ``source`` selects the upload flow and requires a ``multipart/form-data`` body with a
      ``content`` part holding a gzip-compressed tar of the skill directory.
    - A remote ``source`` requires a plain JSON body with no content.

    Args:
        registration: Parsed request metadata.
        content: Stream of the ``content`` part, or ``None`` when the request has none.
        multipart: Whether the request body was ``multipart/form-data``.

    Returns:
        The committed ``SkillVersion``. Every rejection happens before a version row exists.
    """
    _validate_metadata(registration)
    if registration.source is None:
        if not multipart or content is None:
            raise MlflowException.invalid_parameter_value(
                "A registration without a 'source' uploads the skill content and requires a "
                "multipart/form-data body with a 'content' part. To register a remote skill, "
                "set 'source' to its git, oci, or zip location."
            )
        return _register_uploaded(registration, content)
    if multipart or content is not None:
        raise MlflowException.invalid_parameter_value(
            "A registration with a remote 'source' must use an application/json body; it "
            "cannot also carry uploaded content. Omit 'source' to upload content instead."
        )
    return _register_remote(registration)


def _validate_metadata(registration: SkillVersionRegistration) -> None:
    if not registration.name:
        raise MlflowException.invalid_parameter_value(
            "'name' must be provided explicitly. The MLflow SDK and CLI read it from SKILL.md; "
            "the server never reads content to derive it."
        )
    _validate_skill_name(registration.name)
    _validate_organization_name(registration.organization)
    if registration.status not in _REGISTRATION_STATUSES:
        raise MlflowException.invalid_parameter_value(
            f"A skill version can be registered as 'active' or 'draft', got "
            f"{registration.status!r}."
        )
    digest = registration.digest
    if digest is not None and (
        not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None
    ):
        raise MlflowException.invalid_parameter_value(
            "'digest' must be a SHA-256 digest of 64 lowercase hex characters."
        )
    if registration.source_type in (SkillSourceType.MLFLOW.value, SkillSourceType.ASSEMBLED.value):
        raise MlflowException.invalid_parameter_value(
            f"'source_type' {registration.source_type!r} is set by the server from the creation "
            "flow and cannot be submitted."
        )


def _register_remote(registration: SkillVersionRegistration) -> SkillVersion:
    from mlflow.server.handlers import _get_tracking_store

    source = registration.source
    if not isinstance(source, str) or is_local_path(source.strip()):
        raise MlflowException.invalid_parameter_value(
            f"'source' must be a remote git, oci, or zip location, got {source!r}. The server "
            "never reads a path on its own filesystem; local content is uploaded instead."
        )
    if (source_type := registration.source_type) is None:
        resolved = resolve_source_type(source, ref=registration.ref, subpath=registration.subpath)
    else:
        if source_type not in _CLIENT_SOURCE_TYPES:
            raise MlflowException.invalid_parameter_value(
                f"Unknown 'source_type' {source_type!r}; expected one of "
                f"{list(_CLIENT_SOURCE_TYPES)}."
            )
        scheme_type = _type_named_by_scheme(source)
        if scheme_type is not None and scheme_type != source_type:
            raise MlflowException.invalid_parameter_value(
                f"'source_type' {source_type!r} contradicts the {scheme_type!r} scheme of "
                f"'source' {source!r}."
            )
        if source_type != SkillSourceType.GIT.value and registration.ref is not None:
            raise MlflowException.invalid_parameter_value("'ref' applies to git sources only.")
        fields = {"subpath": registration.subpath}
        if source_type == SkillSourceType.GIT.value:
            resolved = resolve_source_type(GitSource(url=source, ref=registration.ref, **fields))
        elif source_type == SkillSourceType.OCI.value:
            resolved = resolve_source_type(OCISource(image=source, **fields))
        else:
            resolved = resolve_source_type(ZipSource(url=source, **fields))
    return _get_tracking_store().create_skill_version(
        name=registration.name,
        organization=registration.organization,
        source_type=resolved.source_type.value,
        source=resolved.source,
        ref=resolved.ref,
        subpath=resolved.subpath,
        digest=registration.digest,
        status=registration.status,
    )


def _type_named_by_scheme(source: str) -> str | None:
    lowered = source.strip().lower()
    if lowered.startswith(OCI_SCHEME):
        return SkillSourceType.OCI.value
    if lowered.startswith("git://"):
        return SkillSourceType.GIT.value
    return None


def _register_uploaded(registration: SkillVersionRegistration, content: BinaryIO) -> SkillVersion:
    from mlflow.server.handlers import _get_tracking_store

    for field in ("source_type", "ref", "subpath"):
        if getattr(registration, field) is not None:
            raise MlflowException.invalid_parameter_value(
                f"'{field}' does not apply to uploaded content; the whole uploaded tree is the "
                "skill and the server records where it stored it."
            )
    require_artifact_serving()
    limit = get_max_decompressed_size()
    with tempfile.TemporaryDirectory(prefix="mlflow-skill-upload-") as tmp:
        archive = Path(tmp) / "content.tar.gz"
        tree = Path(tmp) / "tree"
        _spool(content, archive, max_bytes=limit + _UPLOAD_SIZE_SLACK)
        # Rejects unsafe paths, links and special files, and oversize or malformed archives;
        # nothing has been written to artifact storage yet.
        extract_skill_archive(archive, tree, max_bytes=limit)
        if not collect_tree(tree):
            raise MlflowException.invalid_parameter_value(
                "The uploaded skill content contains no files."
            )
        # A path no other upload can have, so the bytes of a failed or crashed attempt are
        # never picked up by another version.
        artifact_path = new_skill_upload_path(registration.name, registration.organization)
        try:
            store_skill_tree(tree, artifact_path)
        except Exception:
            # No row was attempted for this path, so removing the partial write is safe.
            delete_artifact_tree_best_effort(artifact_path)
            raise
    # The row is committed only now that its content is in place, so a version is complete
    # the moment it exists. A version-number collision is retried inside the store without
    # rewriting the content.
    try:
        return _get_tracking_store().create_skill_version(
            name=registration.name,
            organization=registration.organization,
            source_type=SkillSourceType.MLFLOW.value,
            source=to_artifact_uri(artifact_path),
            digest=registration.digest,
            status=registration.status,
        )
    except MlflowException as e:
        if e.error_code in _DEFINITE_REJECTIONS:
            delete_artifact_tree_best_effort(artifact_path)
        raise


def _spool(content: BinaryIO, target: Path, *, max_bytes: int) -> None:
    received = 0
    with open(target, "wb") as out:
        while chunk := content.read(_UPLOAD_CHUNK_SIZE):
            received += len(chunk)
            if received > max_bytes:
                raise MlflowException.invalid_parameter_value(
                    f"The uploaded skill content exceeds the size limit of {max_bytes} bytes."
                )
            out.write(chunk)
    if received == 0:
        raise MlflowException.invalid_parameter_value("The 'content' part is empty.")
