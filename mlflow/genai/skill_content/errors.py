from __future__ import annotations

import re

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

# Matches the ``user:password@`` part of a URL so credentials never reach an error message.
_USERINFO_PATTERN = re.compile(r"(?<=://)[^/@\s]+@")
# Matches the user part of an scp-style git URL such as ``git@host:org/repo.git``.
_SCP_USERINFO_PATTERN = re.compile(r"(?<![\w/])[^\s/:@]+:[^\s/@]+@(?=[\w.-]+:)")


def redact_credentials(text: str) -> str:
    """Remove URL userinfo (``user:password@``) from ``text`` so it is safe to surface."""
    text = _USERINFO_PATTERN.sub("***@", text)
    return _SCP_USERINFO_PATTERN.sub("***@", text)


def invalid_content(message: str) -> MlflowException:
    """Error for content that violates the skill content rules (paths, entry types, size)."""
    return MlflowException(redact_credentials(message), error_code=INVALID_PARAMETER_VALUE)


def source_unavailable(source: str, detail: str) -> MlflowException:
    """Error for a source that could not be fetched, preserving the underlying reason."""
    return MlflowException(
        f"Failed to fetch skill content from '{redact_credentials(source)}': "
        f"{redact_credentials(detail)}",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
