"""Upload a file to GitHub's user-attachments store and return the URL it serves from."""

from __future__ import annotations

import http.client
import json
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


class UploadFailed(Exception):
    """Raised when one asset does not reach the store; the message says why.

    ``status`` carries the HTTP code when the endpoint answered, so a caller can tell
    a dead credential (401) from a fault that only affects the file at hand.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def max_bytes(name: str) -> int:
    return MAX_VIDEO_BYTES if is_video(name) else MAX_IMAGE_BYTES


def upload_asset(path: Path, repository_id: str, token: str) -> str:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise UploadFailed(f"{path.name}: unsupported extension")

    size = path.stat().st_size
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
            raise UploadFailed(
                f"{TOKEN_ENV} was rejected (401); it may have expired", status=401
            ) from e
        raise UploadFailed(f"{path.name}: {e}", status=e.code) from e
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise UploadFailed(f"{path.name}: {e}") from e

    match body:
        case {"url": str(url)} if url:
            return url
        case _:
            raise UploadFailed(f"{path.name}: response carried no url")
