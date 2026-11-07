from typing import Any

from mlflow.utils.validation import GATEWAY_DESTINATION_KEYS, GATEWAY_REGION_KEYS


async def prepare_litellm_kwargs(
    payload: dict[str, Any], auth_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate caller-controlled options before merging the endpoint's configuration."""
    from fastapi import HTTPException

    from mlflow.gateway.ssrf import (
        GatewaySSRFProtectionError,
        assert_public_upstream_host,
        upstream_protection_enabled,
    )

    if caller_keys := sorted((GATEWAY_DESTINATION_KEYS | GATEWAY_REGION_KEYS) & set(payload)):
        raise HTTPException(
            status_code=400,
            detail=f"The request must not set {', '.join(map(repr, caller_keys))}: the "
            "upstream destination is fixed by the endpoint configuration.",
        )
    kwargs = payload | (auth_config or {})
    if not upstream_protection_enabled():
        return kwargs
    if base_url := kwargs.pop("base_url", None):
        kwargs["api_base"] = base_url
    if stored_keys := sorted((GATEWAY_DESTINATION_KEYS - {"api_base"}) & set(kwargs)):
        raise HTTPException(
            status_code=502,
            detail=f"The endpoint's secret sets {', '.join(map(repr, stored_keys))}, which "
            "is not supported: only 'api_base' may choose the upstream.",
        )
    if api_base := kwargs.get("api_base"):
        try:
            await assert_public_upstream_host(api_base)
        except GatewaySSRFProtectionError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
    return kwargs
