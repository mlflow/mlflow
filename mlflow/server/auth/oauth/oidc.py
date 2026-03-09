import hashlib
import logging
import secrets
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
from cachetools import TTLCache
from flask import redirect, request

from mlflow.server.auth.oauth.config import OIDCProviderConfig

_logger = logging.getLogger(__name__)

# Cache OIDC discovery metadata for 1 hour
_discovery_cache: TTLCache = TTLCache(maxsize=32, ttl=3600)

# Cache JWKS for 1 hour
_jwks_cache: TTLCache = TTLCache(maxsize=32, ttl=3600)


def _fetch_discovery(discovery_url: str) -> dict[str, str]:
    if discovery_url in _discovery_cache:
        return _discovery_cache[discovery_url]

    resp = requests.get(discovery_url, timeout=10)
    resp.raise_for_status()
    metadata = resp.json()
    _discovery_cache[discovery_url] = metadata
    return metadata


def _get_endpoints(provider: OIDCProviderConfig) -> dict[str, str]:
    if provider.discovery_url:
        metadata = _fetch_discovery(provider.discovery_url)
        return {
            "authorization_endpoint": metadata["authorization_endpoint"],
            "token_endpoint": metadata["token_endpoint"],
            "userinfo_endpoint": metadata.get("userinfo_endpoint", ""),
            "jwks_uri": metadata["jwks_uri"],
            "end_session_endpoint": metadata.get("end_session_endpoint", ""),
            "issuer": metadata["issuer"],
        }
    return {
        "authorization_endpoint": provider.auth_url,
        "token_endpoint": provider.token_url,
        "userinfo_endpoint": provider.userinfo_url,
        "jwks_uri": provider.jwks_uri,
        "end_session_endpoint": provider.end_session_endpoint,
        "issuer": "",
    }


def _fetch_jwks(jwks_uri: str) -> dict[str, object]:
    if jwks_uri in _jwks_cache:
        return _jwks_cache[jwks_uri]

    resp = requests.get(jwks_uri, timeout=10)
    resp.raise_for_status()
    jwks = resp.json()
    _jwks_cache[jwks_uri] = jwks
    return jwks


def _generate_pkce() -> tuple[str, str]:
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def start_oidc_flow(provider: OIDCProviderConfig, session_manager, next_url: str = "/"):
    endpoints = _get_endpoints(provider)

    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    code_verifier, code_challenge = _generate_pkce()

    session_manager.store_oauth_state(
        state=state,
        code_verifier=code_verifier,
        nonce=nonce,
        provider_name=provider.name,
        redirect_after_login=next_url,
    )

    redirect_uri = provider.redirect_uri
    if not redirect_uri:
        redirect_uri = f"{request.scheme}://{request.host}/auth/callback"

    params = {
        "response_type": "code",
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "scope": provider.scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "nonce": nonce,
    }

    if provider.extra_auth_params:
        for pair in provider.extra_auth_params.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k.strip()] = v.strip()

    auth_url = f"{endpoints['authorization_endpoint']}?{urlencode(params)}"
    return redirect(auth_url)


def handle_oidc_callback(oauth_config, session_manager, provisioner):
    from authlib.jose import JsonWebKey, jwt

    code = request.args.get("code")
    state = request.args.get("state")
    if error := request.args.get("error"):
        error_desc = request.args.get("error_description", "Unknown error")
        _logger.error("OIDC callback error: %s - %s", error, error_desc)
        return redirect(f"/auth/login?error={error}")

    if not code or not state:
        return redirect("/auth/login?error=missing_params")

    # Retrieve and validate state
    state_data = session_manager.retrieve_and_delete_oauth_state(state)
    if not state_data:
        return redirect("/auth/login?error=invalid_state")

    provider_name = state_data["provider_name"]
    provider = oauth_config.oidc_providers.get(provider_name)
    if not provider:
        return redirect("/auth/login?error=unknown_provider")

    endpoints = _get_endpoints(provider)

    redirect_uri = provider.redirect_uri
    if not redirect_uri:
        redirect_uri = f"{request.scheme}://{request.host}/auth/callback"

    # Exchange code for tokens
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": provider.client_id,
        "client_secret": provider.client_secret,
        "code_verifier": state_data["code_verifier"],
    }

    token_resp = requests.post(endpoints["token_endpoint"], data=token_data, timeout=10)
    if not token_resp.ok:
        _logger.error("Token exchange failed: %s", token_resp.text)
        return redirect("/auth/login?error=token_exchange_failed")

    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    id_token_raw = tokens.get("id_token", "")
    refresh_token = tokens.get("refresh_token", "")
    expires_in = tokens.get("expires_in", 3600)

    # Validate and decode id_token
    if not id_token_raw:
        return redirect("/auth/login?error=no_id_token")

    jwks_data = _fetch_jwks(endpoints["jwks_uri"])
    jwk_set = JsonWebKey.import_key_set(jwks_data)

    claims = jwt.decode(id_token_raw, jwk_set)

    # Validate claims
    expected_aud = provider.expected_audience or provider.client_id

    aud = claims.get("aud", "")
    if isinstance(aud, list):
        if expected_aud not in aud:
            return redirect("/auth/login?error=invalid_audience")
    elif aud != expected_aud:
        return redirect("/auth/login?error=invalid_audience")

    if endpoints.get("issuer"):
        if claims.get("iss") != endpoints["issuer"]:
            return redirect("/auth/login?error=invalid_issuer")

    stored_nonce = state_data["nonce"]
    if claims.get("nonce") != stored_nonce:
        return redirect("/auth/login?error=invalid_nonce")

    # Extract user info
    username = claims.get(provider.username_claim, "")
    email = claims.get(provider.email_claim, "")
    display_name = claims.get(provider.name_claim, "")
    groups = claims.get(provider.groups_claim, [])
    if isinstance(groups, str):
        groups = [groups]

    if not username:
        # Fall back to userinfo endpoint
        if endpoints.get("userinfo_endpoint"):
            userinfo_resp = requests.get(
                endpoints["userinfo_endpoint"],
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            if userinfo_resp.ok:
                userinfo = userinfo_resp.json()
                username = userinfo.get(provider.username_claim, "")
                email = email or userinfo.get(provider.email_claim, "")
                display_name = display_name or userinfo.get(provider.name_claim, "")
                if not groups:
                    groups = userinfo.get(provider.groups_claim, [])

    if not username:
        return redirect("/auth/login?error=no_username")

    # Provision user
    user_id, is_admin = provisioner.provision_user(
        username=username,
        provider_config=provider,
        groups=groups,
    )

    # Calculate token expiry
    token_expiry = datetime.now(timezone.utc)
    if expires_in:
        token_expiry = token_expiry + timedelta(seconds=expires_in)

    # Create session
    id_token_claims = {
        "username": username,
        "email": email,
        "display_name": display_name,
        "groups": groups,
        "is_admin": is_admin,
    }

    session_id = session_manager.create_session(
        user_id=user_id,
        provider=f"oidc:{provider_name}",
        access_token=access_token,
        refresh_token=refresh_token,
        id_token_claims=id_token_claims,
        token_expiry=token_expiry,
        ip_address=request.remote_addr or "",
        user_agent=request.headers.get("User-Agent", "")[:512],
    )

    from mlflow.server.auth.oauth.audit import log_login

    log_login(username, f"oidc:{provider_name}", request.remote_addr or "")

    redirect_url = state_data.get("redirect_after_login", "/")
    response = redirect(redirect_url)
    session_manager.set_session_cookie(response, session_id)
    return response


def refresh_access_token(
    provider: OIDCProviderConfig, refresh_token: str
) -> dict[str, str | int] | None:
    endpoints = _get_endpoints(provider)

    token_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": provider.client_id,
        "client_secret": provider.client_secret,
    }

    resp = requests.post(endpoints["token_endpoint"], data=token_data, timeout=10)
    if not resp.ok:
        _logger.warning("Token refresh failed: %s", resp.text)
        return None

    tokens = resp.json()
    return {
        "access_token": tokens.get("access_token", ""),
        "refresh_token": tokens.get("refresh_token", refresh_token),
        "expires_in": tokens.get("expires_in", 3600),
    }


def validate_bearer_token(token: str, oauth_config) -> dict[str, object] | None:
    from authlib.jose import JsonWebKey, jwt
    from authlib.jose.errors import JoseError

    for provider in oauth_config.oidc_providers.values():
        if not provider.enabled:
            continue

        endpoints = _get_endpoints(provider)
        jwks_uri = endpoints.get("jwks_uri")
        if not jwks_uri:
            continue

        try:
            jwks_data = _fetch_jwks(jwks_uri)
            jwk_set = JsonWebKey.import_key_set(jwks_data)
            claims = jwt.decode(token, jwk_set)

            # Validate audience
            expected_aud = provider.expected_audience or provider.client_id
            aud = claims.get("aud", "")
            if isinstance(aud, list):
                if expected_aud not in aud:
                    continue
            elif aud != expected_aud:
                continue

            # Validate issuer
            if endpoints.get("issuer") and claims.get("iss") != endpoints["issuer"]:
                continue

            # Check expiry
            if exp := claims.get("exp"):
                now = datetime.now(timezone.utc).timestamp()
                if now > exp + provider.clock_skew_seconds:
                    continue

            username = claims.get(provider.username_claim, "")
            if not username:
                continue

            return {
                "username": username,
                "email": claims.get(provider.email_claim, ""),
                "groups": claims.get(provider.groups_claim, []),
                "provider": f"oidc:{provider.name}",
            }
        except (JoseError, Exception):
            continue

    return None
