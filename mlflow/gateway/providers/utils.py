import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncGenerator
from urllib.parse import urlparse, urlunparse

from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
)
from mlflow.utils.uri import append_to_uri_path

# Accumulates the total time (ms) spent waiting for provider HTTP responses in the current
# request context. Reset to 0.0 at the start of each request by the gateway timing middleware
# (add_gateway_timing_middleware in fastapi_app.py).
provider_call_duration_ms: ContextVar[float] = ContextVar("provider_call_duration_ms", default=0.0)

# Request gzip/deflate only so upstream never sends Brotli; aiohttp fails to decode
# Content-Encoding: br without the optional brotli package.
SUPPORTED_ACCEPT_ENCODING = "gzip, deflate, identity"


@asynccontextmanager
async def _aiohttp_post(headers: dict[str, str], base_url: str, path: str, payload: dict[str, Any]):
    import aiohttp

    # Drop any client Accept-Encoding (any casing) so we send only one value; otherwise
    # aiohttp may send both and upstream can respond with Brotli, which is not supported.
    request_headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
    request_headers["Accept-Encoding"] = SUPPORTED_ACCEPT_ENCODING
    url = append_to_uri_path(base_url, path)
    async with aiohttp.ClientSession(headers=request_headers) as session:
        timeout = aiohttp.ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS)
        async with session.post(url, json=payload, timeout=timeout) as response:
            yield response


async def send_request(headers: dict[str, str], base_url: str, path: str, payload: dict[str, Any]):
    """
    Send an HTTP request to a specific URL path with given headers and payload.

    Args:
        headers: The headers to include in the request.
        base_url: The base URL where the request will be sent.
        path: The specific path of the URL to which the request will be sent.
        payload: The payload (or data) to be included in the request.

    Returns:
        The server's response as a JSON object.

    Raises:
        HTTPException if the HTTP request fails.
    """
    import aiohttp
    from fastapi import HTTPException

    start = time.perf_counter()
    try:
        async with _aiohttp_post(headers, base_url, path, payload) as response:
            content_type = response.headers.get("Content-Type")
            if content_type and "application/json" in content_type:
                js = await response.json()
            elif content_type and "text/plain" in content_type:
                js = {"message": await response.text()}
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"The returned data type from the route service is not supported. "
                    f"Received content type: {content_type}",
                )
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                detail = js.get("error", {}).get("message", e.message) if "error" in js else js
                raise HTTPException(status_code=e.status, detail=detail)
    finally:
        # Record full provider HTTP time for non-streaming, even when raising.
        provider_call_duration_ms.set(
            provider_call_duration_ms.get() + (time.perf_counter() - start) * 1000
        )
    return js


async def send_stream_request(
    headers: dict[str, str], base_url: str, path: str, payload: dict[str, Any]
) -> AsyncGenerator[bytes, None]:
    """
    Send a streaming HTTP request to a specific URL path with given headers and payload.

    Args:
        headers: The headers to include in the request.
        base_url: The base URL where the request will be sent.
        path: The specific path of the URL to which the request will be sent.
        payload: The payload (or data) to be included in the request.

    Yields:
        Bytes from the server's streaming response.

    Raises:
        HTTPException if the HTTP request fails.
    """
    import aiohttp
    from fastapi import HTTPException

    async with _aiohttp_post(headers, base_url, path, payload) as response:
        try:
            response.raise_for_status()
        except aiohttp.ClientResponseError as e:
            try:
                error_body = await response.json()
                detail = error_body.get("error", {}).get("message", e.message)
            except Exception:
                detail = e.message
            raise HTTPException(status_code=e.status, detail=detail)

        async for line in response.content:
            yield line


def proxy_root_url(base_url: str) -> str:
    """Return the root URL for raw proxy calls by stripping the last path segment.

    Provider base URLs typically include a versioned path suffix (e.g. ``/v1``).
    Raw proxy callers pass the full API path (e.g. ``v1/messages``), so appending
    that to the base URL would produce a double-versioned URL like
    ``https://api.anthropic.com/v1/v1/messages``.  Stripping the last segment
    returns the true API root (``https://api.anthropic.com``) so the caller's
    path is appended correctly.

    Examples:
        https://api.anthropic.com/v1      → https://api.anthropic.com
        https://api.openai.com/v1         → https://api.openai.com
        https://api.groq.com/openai/v1    → https://api.groq.com/openai
    """
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    new_path = path.rsplit("/", 1)[0] if "/" in path.lstrip("/") else ""
    return urlunparse(parsed._replace(path=new_path))


_STREAMING_CONTENT_TYPES = frozenset({"text/event-stream", "application/x-ndjson"})


async def send_proxy_request(
    headers: dict[str, str],
    base_url: str,
    path: str,
    payload: dict[str, Any],
) -> AsyncGenerator[dict[str, Any] | bytes, None]:
    """
    Async generator for raw proxy requests that auto-detects streaming from Content-Type.

    Yields a metadata dict as the first item:
        {"content_type": str, "is_streaming": bool}

    For streaming responses (``text/event-stream``, ``application/x-ndjson``), then yields
    raw response bytes. For non-streaming responses, yields a single parsed JSON dict.

    Callers should consume all items or explicitly call ``aclose()`` to release the
    underlying HTTP connection.
    """
    import aiohttp
    from fastapi import HTTPException

    start = time.perf_counter()
    try:
        async with _aiohttp_post(headers, base_url, path, payload) as response:
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                try:
                    error_body = await response.json()
                    detail = error_body.get("error", {}).get("message", e.message)
                except Exception:
                    detail = e.message
                raise HTTPException(status_code=e.status, detail=detail)

            content_type = (response.headers or {}).get("Content-Type", "application/json")
            is_streaming = content_type.split(";")[0].strip() in _STREAMING_CONTENT_TYPES

            yield {"content_type": content_type, "is_streaming": is_streaming}

            if is_streaming:
                async for chunk in response.content:
                    yield chunk
            elif "application/json" in content_type:
                yield await response.json()
            elif "text/plain" in content_type:
                yield {"message": await response.text()}
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Unsupported Content-Type from upstream proxy: {content_type}",
                )
    finally:
        provider_call_duration_ms.set(
            provider_call_duration_ms.get() + (time.perf_counter() - start) * 1000
        )


def rename_payload_keys(payload: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Rename payload keys based on the specified mapping. If a key is not present in the
    mapping, the key and its value will remain unchanged.

    Args:
        payload: The original dictionary to transform.
        mapping: A dictionary where each key-value pair represents a mapping from the old
            key to the new key.

    Returns:
        A new dictionary containing the transformed keys.

    """
    return {mapping.get(k, k): v for k, v in payload.items()}
