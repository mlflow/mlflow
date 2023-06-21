from typing import Dict, Any
import aiohttp
from fastapi import HTTPException


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
        url = "/".join([base_url.rstrip("/"), path.lstrip("/")])
        async with session.post(url, json=payload) as response:
            js = await response.json()
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                detail = js.get("error", {}).get("message", e.message)
                raise HTTPException(status_code=e.status, detail=detail)
            return js


def make_payload(payload: Dict[str, Any], mapping: Dict[str, str]):
    """
    Transform a payload based on a provided mapping.

    :param payload: The original payload to transform.
    :param mapping: A dictionary where each key-value pair represents a mapping from the old
                    key to the new key.
    :return: A new dictionary containing the transformed payload.
    """
    payload = payload.copy()
    for k1, k2 in mapping.items():
        if v := payload.pop(k1, None):
            payload[k2] = v
    return {k: v for k, v in payload.items() if v is not None and v != []}
