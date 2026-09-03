from __future__ import annotations

from pathlib import Path

import requests

from mlflow.genai.skill_content.archive import extract_zip_archive
from mlflow.genai.skill_content.errors import invalid_content, source_unavailable

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 60


def download_with_budget(url: str, target: Path, *, max_bytes: int) -> Path:
    """
    Stream ``url`` to ``target``, failing once more than ``max_bytes`` have been received.

    No credentials are attached: ZIP sources are public by policy. The byte budget is enforced
    on the wire so a hostile or oversized download is cut off rather than buffered.
    """
    try:
        with requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            if response.status_code >= 400:
                raise source_unavailable(url, f"HTTP {response.status_code} {response.reason}")
            received = 0
            with open(target, "wb") as out:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    received += len(chunk)
                    if received > max_bytes:
                        raise invalid_content(
                            f"Download from '{url}' exceeds the skill content size limit of "
                            f"{max_bytes} bytes."
                        )
                    out.write(chunk)
    except requests.RequestException as e:
        raise source_unavailable(url, str(e))
    return target


def fetch_zip(url: str, dest: Path, *, max_bytes: int) -> Path:
    """Download the ZIP archive at ``url`` and extract it safely into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "source.zip"
    download_with_budget(url, archive, max_bytes=max_bytes)
    extract_zip_archive(archive, dest, max_bytes=max_bytes)
    archive.unlink(missing_ok=True)
    return dest
