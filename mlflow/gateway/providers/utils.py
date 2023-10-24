from typing import Any, Dict

import aiohttp

from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
)
from mlflow.utils.uri import append_to_uri_path


async def send_request(headers: Dict[str, str], base_url: str, path: str, payload: Dict[str, Any]):
    """
    Send an HTTP request to a specific URL path with given headers and payload.

    :param headers: The headers to include in the request.
    :param base_url: The base URL where the request will be sent.
    :param path: The specific path of the URL to which the request will be sent.
    :param payload: The payload (or data) to be included in the request.
    :return: The server's response as a JSON object.
    :raise: HTTPException if the HTTP request fails.
    """
    from fastapi import HTTPException

    async with aiohttp.ClientSession(headers=headers) as session:
        url = append_to_uri_path(base_url, path)
        timeout = aiohttp.ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS)
        async with session.post(url, json=payload, timeout=timeout) as response:
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
            return js


def rename_payload_keys(payload: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Rename payload keys based on the specified mapping. If a key is not present in the
    mapping, the key and its value will remain unchanged.

    :param payload: The original dictionary to transform.
    :param mapping: A dictionary where each key-value pair represents a mapping from the old
                    key to the new key.
    :return: A new dictionary containing the transformed keys.
    """
    return {mapping.get(k, k): v for k, v in payload.items()}
