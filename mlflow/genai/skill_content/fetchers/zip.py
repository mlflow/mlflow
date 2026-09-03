from __future__ import annotations

from pathlib import Path

import requests

from mlflow.genai.skill_content.archive import extract_zip_archive
from mlflow.genai.skill_content.errors import (
    error_code_for_http_status,
    invalid_content,
    source_unavailable,
)
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 60


def _no_auth(request):
    # Supplying an auth handler stops `requests` from attaching ~/.netrc or URL credentials,
    # while leaving proxy and CA environment handling intact.
    return request


def download_with_budget(url: str, target: Path, *, max_bytes: int) -> Path:
    """
    Stream ``url`` to ``target``, failing once more than ``max_bytes`` have been received.

    No credentials are attached, not even ambient ``~/.netrc`` entries: ZIP sources are public
    by policy. The byte budget is enforced on the wire so a hostile or oversized download is
    cut off rather than buffered.
    """
    session = requests.Session()
    try:
        with session.get(
            url, stream=True, timeout=_REQUEST_TIMEOUT_SECONDS, auth=_no_auth
        ) as response:
            if response.status_code >= 400:
                raise source_unavailable(
                    url,
                    f"HTTP {response.status_code} {response.reason}",
                    error_code=error_code_for_http_status(response.status_code),
                )
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise invalid_content(
                    f"Download from '{url}' is {declared} bytes, which exceeds the skill "
                    f"content size limit of {max_bytes} bytes."
                )
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
        raise source_unavailable(url, str(e), error_code=TEMPORARILY_UNAVAILABLE)
    finally:
        session.close()
    return target


def fetch_zip(url: str, dest: Path, *, max_bytes: int, subpath: str | None = None) -> Path:
    """Download the ZIP archive at ``url`` and extract it (or just ``subpath``) into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "source.zip"
    download_with_budget(url, archive, max_bytes=max_bytes)
    extract_zip_archive(archive, dest, max_bytes=max_bytes, subpath=subpath)
    archive.unlink(missing_ok=True)
    return dest
