from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.provider_filter import normalize_provider_name

_CREDENTIAL_BOUND_AUTH_CONFIG_KEYS = ("api_base",)
_AZURE_OPENAI_API_TYPES = {"azure", "azuread"}
_AZURE_OPENAI_SCOPE = "azure-openai"


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


def _normalize_gateway_provider_scope(
    provider: str | None,
    auth_config: Mapping[str, Any] | None = None,
) -> str | None:
    if not isinstance(provider, str) or not provider.strip():
        return None

    provider_name = normalize_provider_name(provider.strip().lower())
    auth_config = auth_config or {}
    api_type = str(auth_config.get("api_type", "")).lower()

    if provider_name == "azure" or (
        provider_name == "openai" and api_type in _AZURE_OPENAI_API_TYPES
    ):
        return _AZURE_OPENAI_SCOPE

    return provider_name


def validate_gateway_secret_has_provider_scope(provider: str | None) -> None:
    if _normalize_gateway_provider_scope(provider):
        return

    raise MlflowException(
        "Gateway secret provider is required.",
        error_code=INVALID_PARAMETER_VALUE,
    )


def validate_gateway_secret_provider_scope(
    secret_provider: str | None,
    requested_provider: str | None,
    auth_config: Mapping[str, Any] | None = None,
) -> None:
    secret_scope = _normalize_gateway_provider_scope(secret_provider, auth_config)
    requested_scope = _normalize_gateway_provider_scope(requested_provider, auth_config)

    if not secret_scope:
        raise MlflowException(
            "Gateway secret has no provider scope and cannot be used. "
            "Create a new secret for the requested provider.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not requested_scope:
        raise MlflowException(
            "Requested gateway provider is required.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if secret_scope == requested_scope:
        return

    raise MlflowException(
        f"Gateway secret provider '{secret_provider}' cannot be used with "
        f"provider '{requested_provider}'. Create or select a secret for "
        f"provider '{requested_provider}'.",
        error_code=INVALID_PARAMETER_VALUE,
    )
