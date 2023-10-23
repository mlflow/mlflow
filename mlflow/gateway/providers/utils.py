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
    new_dict = {}
    for key, value in payload.items():
        if key in mapping:
            new_key = mapping[key]
            new_value = value
            if "." in new_key:
                parts = new_key.split(".")
                current = new_dict
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = new_value
            else:
                new_dict[new_key] = new_value
        else:
            new_dict[key] = value
    return new_dict


def get_dict_value_by_path(payload: dict, path: str) -> Any:
    """
    Get the value associated with a given key in a nested dictionary.

    Args:
        payload (dict): The nested dictionary containing the data.
        path (str): The key or path to search for in the dictionary.

    Returns:
        Any: The value associated with the specified key, or None if the key does not exist.

    Raises:
        KeyError: If the specified key or path does not exist in any of the nested dictionaries.
    """
    keys = path.split(".")
    value = payload
    for k in keys:
        if not isinstance(value, dict):
            return None
        value = value[k]
    return value


def dict_contains_nested_path(payload: dict, path: str):
    """
    Check whether a nested key exists in a dictionary.

    Args:
        payload (dict): The dictionary to check.
        path (str): The key or path to search for in the dictionary.

    Returns:
        bool: Whether the specified key or path exists in the dictionary.
    """
    try:
        _ = get_dict_value_by_path(payload, path)
        return True
    except KeyError:
        return False
