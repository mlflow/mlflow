# ruff: noqa: T201
"""Upload a file to GitHub's user-attachments store and return the URL it serves from."""

from __future__ import annotations

import http.client
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UPLOAD_URL = "https://uploads.github.com/user-attachments/assets"
# Read from the environment, never argv: a PAT in a CLI argument is visible in the
# process list for the life of the call.
TOKEN_ENV = "UPLOAD_MEDIA_TOKEN"

# The name's extension must agree with content_type or the endpoint returns 422.
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


class TokenRejected(Exception):
    """Raised on 401: the credential is dead, so every remaining upload would fail too."""


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def max_bytes(name: str) -> int:
    return MAX_VIDEO_BYTES if is_video(name) else MAX_IMAGE_BYTES


def upload_asset(path: Path, repository_id: str, token: str) -> str | None:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        print(f"  skip {path.name}: unsupported extension", file=sys.stderr)
        return None

    size = path.stat().st_size
    if size > (limit := max_bytes(path.name)):
        print(f"  skip {path.name}: {size} bytes exceeds {limit}", file=sys.stderr)
        return None

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
        print(f"  failed {path.name}: {e}", file=sys.stderr)
        if e.code == 401:
            raise TokenRejected(f"{TOKEN_ENV} was rejected (401); it may have expired") from e
        return None
    except (OSError, http.client.HTTPException, ValueError) as e:
        print(f"  failed {path.name}: {e}", file=sys.stderr)
        return None

    match body:
        case {"url": str(url)} if url:
            print(f"  uploaded {path.name} -> {url}")
            return url
        case _:
            print(f"  failed {path.name}: response carried no url", file=sys.stderr)
            return None
