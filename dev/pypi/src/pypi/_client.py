from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from typing import Any, Literal, overload

import aiohttp

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


_cache: dict[str, Package] = {}


def _url(name: str) -> str:
    return f"{_base_url()}/pypi/{name}/json"


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    last_error: object | None = None
    for attempt in range(_RETRIES):
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    raise PyPIError(f"Package not found: {url}")
                if resp.status not in _RETRYABLE_STATUSES and not 200 <= resp.status < 300:
                    raise PyPIError(f"HTTP {resp.status} from {url}: {resp.reason}")
                if 200 <= resp.status < 300:
                    payload = await resp.json()
                    if not isinstance(payload, dict):
                        raise PyPIError(f"Unexpected response from {url}: {type(payload).__name__}")
                    return payload
                last_error = f"HTTP {resp.status}"
        except aiohttp.ClientError as e:
            last_error = e

        if attempt + 1 < _RETRIES:
            await asyncio.sleep(_BACKOFF_BASE * (2**attempt))

    raise PyPIError(f"Failed to fetch {url} after {_RETRIES} attempts: {last_error}")


async def _fetch_one(session: aiohttp.ClientSession, name: str) -> Package:
    if (cached := _cache.get(name)) is not None:
        return cached
    pkg = Package.from_json(await _fetch_json(session, _url(name)))
    _cache[name] = pkg
    return pkg


async def get_package(name: str) -> Package:
    """Fetch package metadata from PyPI. Override the base URL with $PYPI_URL."""
    return (await get_packages([name]))[0]


@overload
async def get_packages(
    names: Iterable[str], *, return_exceptions: Literal[False] = ...
) -> list[Package]: ...
@overload
async def get_packages(
    names: Iterable[str], *, return_exceptions: Literal[True]
) -> list[Package | BaseException]: ...
async def get_packages(
    names: Iterable[str], *, return_exceptions: bool = False
) -> list[Package] | list[Package | BaseException]:
    """Fetch multiple packages concurrently with one shared aiohttp session.

    Set ``return_exceptions=True`` to mirror :func:`asyncio.gather`'s behavior:
    failed fetches return their ``BaseException`` in place of a ``Package`` rather
    than aborting the whole batch.
    """
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(
            *(_fetch_one(session, n) for n in names),
            return_exceptions=return_exceptions,
        )


def clear_cache() -> None:
    _cache.clear()


__all__ = ["PyPIError", "clear_cache", "get_package", "get_packages"]
