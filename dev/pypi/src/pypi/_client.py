from __future__ import annotations

import asyncio
import functools
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any

from pypi._models import Package

_DEFAULT_PYPI_URL = "https://pypi.org"
_TIMEOUT_SECONDS = 10.0
_RETRIES = 3
_BACKOFF_BASE = 0.5

# Transient HTTP statuses worth retrying: rate limits and server-side hiccups.
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class PyPIError(RuntimeError):
    pass


def _base_url() -> str:
    return os.environ.get("PYPI_URL", _DEFAULT_PYPI_URL).rstrip("/")


def _fetch_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise PyPIError(f"Package not found: {url}") from e
            if e.code not in _RETRYABLE_STATUSES:
                raise PyPIError(f"HTTP {e.code} from {url}: {e.reason}") from e
            last_error = e
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as e:
            # URLError wraps socket errors, DNS failures, refused connections.
            last_error = e
        except json.JSONDecodeError as e:
            # Usually means an upstream proxy returned an HTML error page.
            last_error = e
        else:
            if not isinstance(payload, dict):
                raise PyPIError(f"Unexpected response from {url}: {type(payload).__name__}")
            return payload

        if attempt + 1 < _RETRIES:
            time.sleep(_BACKOFF_BASE * (2**attempt))

    raise PyPIError(f"Failed to fetch {url} after {_RETRIES} attempts: {last_error}")


@functools.cache
def get_package(name: str) -> Package:
    """Fetch package metadata from PyPI. Override the base URL with $PYPI_URL."""
    url = f"{_base_url()}/pypi/{name}/json"
    return Package.from_json(_fetch_json(url))


async def aget_package(name: str) -> Package:
    """Async wrapper around :func:`get_package`. Each call runs in a thread."""
    return await asyncio.to_thread(get_package, name)


async def aget_packages(names: Iterable[str]) -> list[Package]:
    """Fetch multiple packages concurrently. Order matches ``names``."""
    return await asyncio.gather(*(aget_package(n) for n in names))


def clear_cache() -> None:
    get_package.cache_clear()


__all__ = ["PyPIError", "aget_package", "aget_packages", "clear_cache", "get_package"]
