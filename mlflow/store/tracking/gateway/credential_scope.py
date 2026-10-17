from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


def load_gateway_auth_config(value: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            raise MlflowException(
                "Invalid gateway secret auth_config JSON.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from e
        return parsed or None
    return dict(value) or None


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
    existing_auth_config: Mapping[str, Any] | None,
    updated_auth_config: Mapping[str, Any] | None,
    *,
    secret_value_provided: bool,
) -> None:
    if updated_auth_config is None or secret_value_provided:
        return

    updated_hostname = _api_base_hostname(updated_auth_config)
    if updated_hostname is None or updated_hostname == _api_base_hostname(existing_auth_config):
        return

    raise MlflowException(
        "Changing the API Base URL hostname requires providing secret_value in the same request.",
        error_code=INVALID_PARAMETER_VALUE,
    )
