"""Upload a file to GitHub's user-attachments store and return the URL it serves from."""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UPLOAD_URL = "https://uploads.github.com/user-attachments/assets"

# The name's extension must agree with content_type or the endpoint returns 422. The
# allowlist is narrower than the formats GitHub documents for attachments: audio is
# refused, so extend this map only against a live 201. svg does upload, but is left out
# until something confirms GitHub renders an svg attachment in markdown.
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}


# Answered for a fault that is not about the file at hand, so every remaining upload in
# the run would fail the same way.
FATAL_STATUSES = frozenset({401, 403, 404, 429})

# The endpoint serves only OAuth tokens, classic PATs, and fine-grained PATs bound to the
# target repository. Actions and App installation tokens are refused whatever their
# permissions, and the refusal is a bare 404, so name the kind rather than leave the
# caller guessing.
TOKEN_KINDS = {
    "gho_": "an OAuth user token",
    "ghp_": "a classic PAT",
    "github_pat_": "a fine-grained PAT",
    "ghu_": "a GitHub App user-to-server token",
    "ghs_": "a GitHub App or Actions token",
    "ghr_": "a refresh token",
}


def describe_token(token: str) -> str:
    for prefix, kind in TOKEN_KINDS.items():
        if token.startswith(prefix):
            return kind
    return "a credential of unrecognized kind"


class UploadFailed(Exception):
    """Raised when one asset does not reach the store; the message says why.

    ``status`` carries the HTTP code when the endpoint answered, and ``fatal`` says
    whether the rest of the run can continue.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def fatal(self) -> bool:
        """Whether retrying the remaining files would fail identically."""
        return self.status in FATAL_STATUSES


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def error_detail(e: urllib.error.HTTPError) -> str:
    """Join what the endpoint said about the refusal.

    The reason phrase names no cause; a 422 explains itself only in the body, and that
    is the status that separates a bad content type from an oversized file.
    """
    try:
        body = json.loads(e.read())
    except (OSError, ValueError):
        return ""

    parts: list[str] = []
    match body:
        case {"message": str(message)}:
            parts.append(message)
    match body:
        case {"errors": [*errors]}:
            for error in errors:
                match error:
                    case {"message": str(message)}:
                        # The size refusal embeds markup meant for the web uploader, and
                        # dropping the tags leaves the whitespace that sat between them.
                        parts.append(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", message)).strip())
                    # Only `code: custom` is documented to carry a message, so name the
                    # field and code rather than collapse to a bare "Validation Failed".
                    case {"code": str(code), "field": str(field)}:
                        parts.append(f"{field}: {code}")
                    case {"code": str(code)}:
                        parts.append(code)
    return "; ".join(dict.fromkeys(parts))


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def max_bytes(name: str) -> int:
    return MAX_VIDEO_BYTES if is_video(name) else MAX_IMAGE_BYTES


def upload_asset(path: Path, repository_id: str, token: str) -> str:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise UploadFailed(f"{path.name}: unsupported extension")

    size = path.stat().st_size
    if size == 0:
        # The endpoint refuses an empty file as 422 "Yowza that's a big file. Try again
        # with a file size less than 10MB", which points at the opposite problem.
        raise UploadFailed(f"{path.name}: the file is empty")
    if size > (limit := max_bytes(path.name)):
        raise UploadFailed(f"{path.name}: {size} bytes exceeds {limit}")

    query = urllib.parse.urlencode({
        "name": path.name,
        "content_type": mime,
        "repository_id": repository_id,
    })
    request = urllib.request.Request(
        f"{UPLOAD_URL}?{query}",
        data=path.read_bytes(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            body = json.load(resp)
    # Must precede OSError: HTTPError is a URLError, which is an OSError.
    except urllib.error.HTTPError as e:
        if e.code == 401:
            # Callers resolve credentials differently, so name none of them here.
            raise UploadFailed(f"{path.name}: the credential was rejected (401)", status=401) from e
        if e.code == 403:
            raise UploadFailed(
                f"{path.name}: 403, most likely a credential that is not scoped to this repository",
                status=403,
            ) from e
        if e.code == 404:
            raise UploadFailed(
                f"{path.name}: 404, so either repository_id={repository_id} does not resolve or "
                f"the endpoint refuses this credential, which is {describe_token(token)}",
                status=404,
            ) from e
        if e.code == 429:
            # Retry-After rides along only on a secondary limit; a primary one explains
            # itself in the body, and the two need different waits.
            after = e.headers.get("Retry-After")
            detail = error_detail(e)
            raise UploadFailed(
                f"{path.name}: rate limited (429)"
                + (f": {detail}" if detail else "")
                + (f"; retry after {after}s" if after else ""),
                status=429,
            ) from e
        detail = error_detail(e)
        raise UploadFailed(
            f"{path.name}: {e.code} {e.reason}" + (f": {detail}" if detail else ""),
            status=e.code,
        ) from e
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise UploadFailed(f"{path.name}: {e}") from e

    match body:
        case {"url": str(url)} if url:
            return url
        case _:
            raise UploadFailed(f"{path.name}: response carried no url")
