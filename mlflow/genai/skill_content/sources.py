from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.genai.skill_content.paths import normalize_subpath

OCI_SCHEME = "oci://"
_GIT_SCHEME = "git://"
# ``mlflow-artifacts:/`` is the form the registry persists for MLflow-stored content. The
# ``runs:/`` and ``models:/`` forms are accepted for fetching (introspection of content that
# already lives in MLflow) but are never what the registry records as a version's source.
_MLFLOW_PREFIXES = ("mlflow-artifacts:/", "runs:/", "models:/")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_SCP_GIT_PATTERN = re.compile(r"^[\w.-]+@[\w.-]+:")
_UNSAFE_REF_PATTERN = re.compile(r"[\s\x00]")

SourceInput = GitSource | OCISource | ZipSource | str


@dataclass(frozen=True)
class ResolvedSource:
    """
    A source pointer with its type settled.

    ``source_type`` is the explicit type carried by a typed source, or the type inferred from a
    plain string. ``source`` is the bare image reference for OCI (no ``oci://`` scheme), the
    URL for Git and ZIP, or the artifact URI for MLflow-stored content; those are the values
    the register flow submits. When ``is_local`` is set, ``source`` is the absolute local path
    to upload and is never persisted; the server records where it stored the content.
    """

    source_type: SkillSourceType
    source: str
    ref: str | None = None
    subpath: str | None = None
    is_local: bool = False


def is_local_path(source: str) -> bool:
    """Whether a plain-string source names a local filesystem path rather than a remote pointer."""
    if _WINDOWS_DRIVE_PATTERN.match(source):
        return True
    if "://" in source or source.startswith(_MLFLOW_PREFIXES):
        return False
    return not _SCP_GIT_PATTERN.match(source)


def _validate_remote_value(value: str, what: str) -> None:
    if value.startswith("-"):
        raise invalid_content(f"{what} must not start with '-': '{value}'.")
    if _UNSAFE_REF_PATTERN.search(value):
        raise invalid_content(f"{what} must not contain whitespace or control characters.")


def _validate_git(url: str, ref: str | None) -> None:
    _validate_remote_value(url, "Git URL")
    if ref is not None:
        if not ref:
            raise invalid_content("Git ref must not be empty.")
        _validate_remote_value(ref, "Git ref")


def _validate_zip(url: str) -> None:
    _validate_remote_value(url, "ZIP URL")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise invalid_content(f"ZIP source must be an http(s) URL, got '{url}'.")
    if parts.username or parts.password:
        raise invalid_content(
            "ZIP sources are fetched without authentication and must be publicly accessible; "
            "remove the credentials from the URL or use a Git or OCI source instead."
        )


def _infer_remote_type(source: str) -> tuple[SkillSourceType, str]:
    """
    Apply the RFC inference table to a plain string: ``oci://`` scheme, ``.git`` suffix or
    ``git://`` scheme, ``.zip`` suffix, or an MLflow artifact URI. Anything else is ambiguous.
    """
    if source.startswith(OCI_SCHEME):
        image = source.removeprefix(OCI_SCHEME)
        if not image:
            raise invalid_content("OCI source must include an image reference after 'oci://'.")
        return SkillSourceType.OCI, image
    if source.startswith(_MLFLOW_PREFIXES):
        return SkillSourceType.MLFLOW, source
    if source.startswith(_GIT_SCHEME):
        return SkillSourceType.GIT, source
    path = urlsplit(source).path
    if path.endswith(".git") or path.endswith(".git/"):
        return SkillSourceType.GIT, source
    if path.endswith(".zip"):
        return SkillSourceType.ZIP, source
    raise invalid_content(
        f"Cannot infer the source type of '{source}'. Pass a typed source (GitSource, "
        "OCISource, or ZipSource) or use a URL that ends in '.git' or '.zip', or starts with "
        "'git://' or 'oci://'."
    )


def resolve_source_type(
    source: SourceInput, *, ref: str | None = None, subpath: str | None = None
) -> ResolvedSource:
    """
    Settle the type, pointer, ref, and subpath of a skill source.

    Typed sources keep their explicit type; ``ref`` and ``subpath`` must then be omitted since
    the object already carries them. A plain string is classified as a local path or has its
    remote type inferred by the RFC table; an external value that cannot be inferred is
    rejected rather than guessed. ``mlflow`` for a local path is flow-derived, never something
    a caller asserts. Git URLs and refs that look like command-line options, and ZIP URLs
    that carry credentials, are rejected.
    """
    if isinstance(source, (GitSource, OCISource, ZipSource)):
        if ref is not None or subpath is not None:
            raise invalid_content(
                "'ref' and 'subpath' must not be passed alongside a typed source; set them on "
                f"the {type(source).__name__} instead."
            )
        if isinstance(source, GitSource):
            _validate_git(source.url, source.ref)
            return ResolvedSource(
                SkillSourceType.GIT,
                source.url,
                ref=source.ref,
                subpath=normalize_subpath(source.subpath),
            )
        if isinstance(source, OCISource):
            image = source.image.removeprefix(OCI_SCHEME)
            if not image:
                raise invalid_content("OCISource image reference must not be empty.")
            _validate_remote_value(image, "OCI image reference")
            return ResolvedSource(
                SkillSourceType.OCI, image, subpath=normalize_subpath(source.subpath)
            )
        _validate_zip(source.url)
        return ResolvedSource(
            SkillSourceType.ZIP, source.url, subpath=normalize_subpath(source.subpath)
        )

    if not isinstance(source, str) or not source.strip():
        raise invalid_content("Skill source must be a typed source or a non-empty string.")
    value = source.strip()
    normalized_subpath = normalize_subpath(subpath)
    if is_local_path(value):
        if ref is not None:
            raise invalid_content("'ref' applies to Git sources only, not to a local path.")
        return ResolvedSource(
            SkillSourceType.MLFLOW,
            str(Path(value).expanduser().resolve()),
            subpath=normalized_subpath,
            is_local=True,
        )
    source_type, pointer = _infer_remote_type(value)
    if ref is not None and source_type != SkillSourceType.GIT:
        raise invalid_content("'ref' applies to Git sources only.")
    if source_type == SkillSourceType.GIT:
        _validate_git(pointer, ref)
    elif source_type == SkillSourceType.ZIP:
        _validate_zip(pointer)
    else:
        _validate_remote_value(pointer, "Source")
    return ResolvedSource(source_type, pointer, ref=ref, subpath=normalized_subpath)
