import logging
import secrets
from pathlib import Path

from flask import redirect, request

_logger = logging.getLogger(__name__)


def _get_saml_settings(provider_config, request_data: dict[str, object]) -> dict[str, object]:
    sp_acs_url = provider_config.sp_acs_url
    if not sp_acs_url:
        sp_acs_url = f"{request_data['https']}://{request_data['http_host']}/auth/saml/acs"

    sp_slo_url = provider_config.sp_slo_url
    if not sp_slo_url:
        sp_slo_url = f"{request_data['https']}://{request_data['http_host']}/auth/saml/slo"

    settings = {
        "strict": True,
        "debug": False,
        "sp": {
            "entityId": provider_config.sp_entity_id,
            "assertionConsumerService": {
                "url": sp_acs_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
            },
            "singleLogoutService": {
                "url": sp_slo_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
        },
        "security": {
            "wantAssertionsSigned": provider_config.want_assertions_signed,
            "authnRequestsSigned": bool(provider_config.sp_cert_file),
        },
    }

    if provider_config.sp_cert_file:
        cert_path = Path(provider_config.sp_cert_file)
        key_path = Path(provider_config.sp_key_file) if provider_config.sp_key_file else None
        settings["sp"]["x509cert"] = cert_path.read_text()
        if key_path:
            settings["sp"]["privateKey"] = key_path.read_text()

    return settings


def _get_request_data() -> dict[str, object]:
    return {
        "https": "on" if request.scheme == "https" else "off",
        "http_host": request.host,
        "script_name": request.path,
        "get_data": request.args.to_dict(),
        "post_data": request.form.to_dict(),
    }


def _load_idp_metadata(provider_config, settings: dict[str, object]) -> dict[str, object]:
    from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser

    if provider_config.idp_metadata_url:
        idp_metadata = OneLogin_Saml2_IdPMetadataParser.parse_remote(
            provider_config.idp_metadata_url
        )
    elif provider_config.idp_metadata_file:
        xml = Path(provider_config.idp_metadata_file).read_text()
        idp_metadata = OneLogin_Saml2_IdPMetadataParser.parse(xml)
    else:
        raise ValueError("Either idp_metadata_url or idp_metadata_file must be configured")

    return OneLogin_Saml2_IdPMetadataParser.merge(settings, idp_metadata)


def start_saml_flow(provider_config, session_manager, next_url: str = "/"):
    from onelogin.saml2.auth import OneLogin_Saml2_Auth

    request_data = _get_request_data()
    settings = _get_saml_settings(provider_config, request_data)
    settings = _load_idp_metadata(provider_config, settings)

    auth = OneLogin_Saml2_Auth(request_data, settings)

    state = secrets.token_urlsafe(32)
    session_manager.store_oauth_state(
        state=state,
        code_verifier="",
        nonce="",
        provider_name=provider_config.name,
        redirect_after_login=next_url,
    )

    sso_url = auth.login(return_to=state)
    return redirect(sso_url)


def handle_saml_acs(oauth_config, session_manager, provisioner):
    from onelogin.saml2.auth import OneLogin_Saml2_Auth

    request_data = _get_request_data()
    relay_state = request.form.get("RelayState", "")

    # Find provider from relay state
    state_data = session_manager.retrieve_and_delete_oauth_state(relay_state)
    if not state_data:
        return redirect("/auth/login?error=invalid_state")

    provider_name = state_data["provider_name"]
    provider = oauth_config.saml_providers.get(provider_name)
    if not provider:
        return redirect("/auth/login?error=unknown_provider")

    settings = _get_saml_settings(provider, request_data)
    settings = _load_idp_metadata(provider, settings)

    auth = OneLogin_Saml2_Auth(request_data, settings)
    auth.process_response()

    if errors := auth.get_errors():
        _logger.error("SAML validation errors: %s", errors)
        return redirect("/auth/login?error=saml_validation_failed")

    if not auth.is_authenticated():
        return redirect("/auth/login?error=saml_not_authenticated")

    attributes = auth.get_attributes()
    name_id = auth.get_nameid()

    # Extract user info from SAML attributes
    username = ""
    if provider.username_attribute:
        vals = attributes.get(provider.username_attribute, [])
        username = vals[0] if vals else ""
    if not username:
        username = name_id

    email = ""
    if provider.email_attribute:
        vals = attributes.get(provider.email_attribute, [])
        email = vals[0] if vals else ""

    groups = []
    if provider.groups_attribute:
        groups = attributes.get(provider.groups_attribute, [])

    if not username:
        return redirect("/auth/login?error=no_username")

    # Provision user
    user_id, is_admin = provisioner.provision_user(
        username=username,
        provider_config=provider,
        groups=groups,
    )

    # Create session
    id_token_claims = {
        "username": username,
        "email": email,
        "groups": groups,
        "is_admin": is_admin,
        "saml_name_id": name_id,
    }

    session_id = session_manager.create_session(
        user_id=user_id,
        provider=f"saml:{provider_name}",
        id_token_claims=id_token_claims,
        ip_address=request.remote_addr or "",
        user_agent=request.headers.get("User-Agent", "")[:512],
    )

    redirect_url = state_data.get("redirect_after_login", "/")
    response = redirect(redirect_url)
    session_manager.set_session_cookie(response, session_id)
    return response


def handle_saml_slo(oauth_config, session_manager):
    from onelogin.saml2.auth import OneLogin_Saml2_Auth

    request_data = _get_request_data()

    # Try each SAML provider
    for provider in oauth_config.saml_providers.values():
        if not provider.enabled:
            continue

        try:
            settings = _get_saml_settings(provider, request_data)
            settings = _load_idp_metadata(provider, settings)
            auth = OneLogin_Saml2_Auth(request_data, settings)

            if request.method == "POST":
                auth.process_slo(delete_session_cb=lambda: None)
            else:
                auth.process_slo(delete_session_cb=lambda: None)

            # Delete local session
            if session_id := session_manager.get_session_id_from_cookie(request):
                session_manager.delete_session(session_id)

            return redirect("/auth/login")
        except Exception:
            _logger.exception("SAML SLO error for provider %s", provider.name)
            continue

    return redirect("/auth/login")
