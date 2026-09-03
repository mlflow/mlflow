from __future__ import annotations

import re

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
    UNAUTHENTICATED,
)

# ``user:password@`` inside a URL.
_USERINFO_PATTERN = re.compile(r"(?<=://)[^/@\s]+@")
# The user part of an scp-style git URL such as ``user:token@host:org/repo.git``.
_SCP_USERINFO_PATTERN = re.compile(r"(?<![\w/])[^\s/:@]+:[^\s/@]+@(?=[\w.-]+:)")
# Bearer / Basic tokens that transport libraries sometimes echo into messages.
_AUTH_TOKEN_PATTERN = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")


def redact_credentials(text: str) -> str:
    """Remove URL userinfo and auth tokens from ``text`` so it is safe to surface."""
    text = _USERINFO_PATTERN.sub("***@", text)
    text = _SCP_USERINFO_PATTERN.sub("***@", text)
    return _AUTH_TOKEN_PATTERN.sub(r"\1 ***", text)


def invalid_content(message: str) -> MlflowException:
    """Error for content that violates the skill content rules (paths, entry types, size)."""
    return MlflowException(redact_credentials(message), error_code=INVALID_PARAMETER_VALUE)


def error_code_for_http_status(status: int) -> int:
    if status == 401:
        return UNAUTHENTICATED
    if status == 403:
        return PERMISSION_DENIED
    if status in (404, 410):
        return RESOURCE_DOES_NOT_EXIST
    if status >= 500 or status == 429:
        return TEMPORARILY_UNAVAILABLE
    return RESOURCE_DOES_NOT_EXIST


def source_unavailable(
    source: str, detail: str, *, error_code: int = RESOURCE_DOES_NOT_EXIST
) -> MlflowException:
    """
    Error for a source that could not be fetched, preserving the underlying reason.

    ``error_code`` distinguishes authentication, permission, availability, and not-found
    failures so callers such as the CLI can map them to distinct exit statuses.
    """
    return MlflowException(
        f"Failed to fetch skill content from '{redact_credentials(source)}': "
        f"{redact_credentials(detail)}",
        error_code=error_code,
    )
