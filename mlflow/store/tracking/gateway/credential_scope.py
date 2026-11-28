from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE


def _api_base_hostname(auth_config: Mapping[str, Any] | None) -> str | None:
    api_base = (auth_config or {}).get("api_base")
    if not isinstance(api_base, str) or not (api_base := api_base.strip()):
        return None
    # Fall back to the original value if the URL has no parsed hostname. This
    # keeps malformed or relative destinations from silently bypassing the check.
    try:
        return urlparse(api_base).hostname or api_base
    except ValueError:
        return api_base


def validate_gateway_secret_update_does_not_retarget_credential(
    existing_auth_config: str | None,
    updated_auth_config: Mapping[str, Any] | None,
    *,
    secret_value_provided: bool,
) -> None:
    if updated_auth_config is None or secret_value_provided:
        return

    updated_hostname = _api_base_hostname(updated_auth_config)
    if updated_hostname is None:
        return

    try:
        parsed_auth_config = json.loads(existing_auth_config) if existing_auth_config else None
    except json.JSONDecodeError as e:
        raise MlflowException(
            "Stored gateway secret auth_config is not a valid JSON object.",
            error_code=INTERNAL_ERROR,
        ) from e

    if existing_auth_config and not isinstance(parsed_auth_config, dict):
        raise MlflowException(
            "Stored gateway secret auth_config is not a valid JSON object.",
            error_code=INTERNAL_ERROR,
        )

    if updated_hostname == _api_base_hostname(parsed_auth_config):
        return

    raise MlflowException(
        "Changing the API Base URL hostname requires providing secret_value in the same request.",
        error_code=INVALID_PARAMETER_VALUE,
    )
