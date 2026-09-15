from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_CREDENTIAL_BOUND_AUTH_CONFIG_KEYS = ("api_base",)


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


def _normalize_bound_auth_config_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return value


def _get_credential_bound_auth_config(auth_config: Mapping[str, Any] | None) -> dict[str, Any]:
    auth_config = auth_config or {}
    return {
        key: _normalize_bound_auth_config_value(auth_config.get(key))
        for key in _CREDENTIAL_BOUND_AUTH_CONFIG_KEYS
    }


def validate_gateway_secret_update_does_not_retarget_credential(
    existing_auth_config: Mapping[str, Any] | None,
    updated_auth_config: Mapping[str, Any] | None,
    *,
    secret_value_provided: bool,
) -> None:
    if updated_auth_config is None or secret_value_provided:
        return

    if _get_credential_bound_auth_config(existing_auth_config) == _get_credential_bound_auth_config(
        updated_auth_config
    ):
        return

    formatted_keys = ", ".join(repr(key) for key in _CREDENTIAL_BOUND_AUTH_CONFIG_KEYS)
    raise MlflowException(
        "Updating credential-bound auth_config field(s) "
        f"{formatted_keys} requires providing a replacement secret_value in the same request.",
        error_code=INVALID_PARAMETER_VALUE,
    )
