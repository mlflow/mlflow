import aiohttp
from typing import Dict, Any
from fastapi import HTTPException

from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
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
    async with aiohttp.ClientSession(headers=headers) as session:
        url = append_to_uri_path(base_url, path)
        timeout = aiohttp.ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS)
        async with session.post(url, json=payload, timeout=timeout) as response:
            js = await response.json()
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                detail = js.get("error", {}).get("message", e.message) if "error" in js else js
                raise HTTPException(status_code=e.status, detail=detail)
            return js


def rename_payload_keys(payload: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Transform the keys in a dictionary based on a provided mapping.

    :param payload: The original dictionary to transform.
    :param mapping: A dictionary where each key-value pair represents a mapping from the old
                    key to the new key.
    :return: A new dictionary containing the transformed keys.
    """
    result = payload.copy()
    for old_key, new_key in mapping.items():
        if old_key in result:
            result[new_key] = result.pop(old_key)
    return result
