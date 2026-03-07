import configparser
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH
from mlflow.server.auth.config import AuthConfig, read_auth_config


@dataclass
class OIDCProviderConfig:
    name: str
    enabled: bool = True
    display_name: str = "Sign in with SSO"
    discovery_url: str = ""
    auth_url: str = ""
    token_url: str = ""
    userinfo_url: str = ""
    jwks_uri: str = ""
    end_session_endpoint: str = ""
    client_id: str = ""
    client_secret: str = ""
    scopes: str = "openid profile email"
    redirect_uri: str = ""
    username_claim: str = "preferred_username"
    email_claim: str = "email"
    groups_claim: str = "groups"
    name_claim: str = "name"
    role_mappings: str = ""
    admin_groups: str = ""
    expected_audience: str = ""
    clock_skew_seconds: int = 30
    extra_auth_params: str = ""


@dataclass
class SAMLProviderConfig:
    name: str
    enabled: bool = False
    display_name: str = "Corporate SSO (SAML)"
    idp_metadata_url: str = ""
    idp_metadata_file: str = ""
    sp_entity_id: str = "mlflow"
    sp_acs_url: str = ""
    sp_slo_url: str = ""
    username_attribute: str = ""
    email_attribute: str = ""
    groups_attribute: str = ""
    role_mappings: str = ""
    admin_groups: str = ""
    sp_cert_file: str = ""
    sp_key_file: str = ""
    want_assertions_signed: bool = True


@dataclass
class ExternalAuthzConfig:
    enabled: bool = False
    endpoint: str = ""
    forward_token: bool = True
    headers: str = ""
    allowed_field: str = "allowed"
    permission_field: str = "permission"
    admin_field: str = "is_admin"
    cache_ttl_seconds: int = 300
    cache_max_size: int = 10000
    timeout_seconds: int = 5
    max_retries: int = 1
    retry_backoff_seconds: float = 0.5
    on_error: Literal["deny", "fallback_to_default", "allow"] = "deny"


@dataclass
class OAuthConfig:
    auth_config: AuthConfig
    session_lifetime_seconds: int = 86400
    idle_timeout_seconds: int = 3600
    session_refresh_threshold_seconds: int = 300
    session_cookie_name: str = "mlflow_session"
    session_cookie_secure: bool = True
    encryption_key: str = ""
    auto_provision_users: bool = True
    allow_basic_auth_fallback: bool = False
    auto_redirect_single_provider: bool = False
    oidc_providers: dict[str, OIDCProviderConfig] = field(default_factory=dict)
    saml_providers: dict[str, SAMLProviderConfig] = field(default_factory=dict)
    external_authz: ExternalAuthzConfig = field(default_factory=ExternalAuthzConfig)

    def get_enabled_providers(self) -> list[dict[str, str]]:
        providers = []
        for name, cfg in self.oidc_providers.items():
            if cfg.enabled:
                providers.append({"name": name, "display_name": cfg.display_name, "type": "oidc"})
        for name, cfg in self.saml_providers.items():
            if cfg.enabled:
                providers.append({"name": name, "display_name": cfg.display_name, "type": "saml"})
        return providers

    def get_provider(self, provider_name: str) -> OIDCProviderConfig | SAMLProviderConfig | None:
        if provider_name in self.oidc_providers:
            return self.oidc_providers[provider_name]
        if provider_name in self.saml_providers:
            return self.saml_providers[provider_name]
        return None

    def parse_role_mappings(self, mappings_str: str) -> dict[str, str]:
        if not mappings_str:
            return {}
        result = {}
        for mapping in mappings_str.split(","):
            mapping = mapping.strip()
            if ":" in mapping:
                group, permission = mapping.rsplit(":", 1)
                result[group.strip()] = permission.strip()
        return result

    def parse_admin_groups(self, admin_groups_str: str) -> list[str]:
        if not admin_groups_str:
            return []
        return [g.strip() for g in admin_groups_str.split(",") if g.strip()]


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: str) -> str:
    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _get_str(config: configparser.ConfigParser, section: str, key: str, fallback: str = "") -> str:
    raw = config.get(section, key, fallback=fallback)
    return _resolve_env_vars(raw)


def _get_oauth_config_path() -> str:
    return MLFLOW_AUTH_CONFIG_PATH.get() or str(
        Path(__file__).parent.parent.joinpath("basic_auth.ini").resolve()
    )


def read_oauth_config() -> OAuthConfig:
    config_path = _get_oauth_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)

    auth_config = read_auth_config()

    oauth_cfg = OAuthConfig(auth_config=auth_config)

    if config.has_section("oauth"):
        oauth_cfg.session_lifetime_seconds = config.getint(
            "oauth", "session_lifetime_seconds", fallback=86400
        )
        oauth_cfg.idle_timeout_seconds = config.getint(
            "oauth", "idle_timeout_seconds", fallback=3600
        )
        oauth_cfg.session_refresh_threshold_seconds = config.getint(
            "oauth", "session_refresh_threshold_seconds", fallback=300
        )
        oauth_cfg.session_cookie_name = _get_str(
            config, "oauth", "session_cookie_name", "mlflow_session"
        )
        oauth_cfg.session_cookie_secure = config.getboolean(
            "oauth", "session_cookie_secure", fallback=True
        )
        oauth_cfg.encryption_key = _get_str(config, "oauth", "encryption_key", "")
        oauth_cfg.auto_provision_users = config.getboolean(
            "oauth", "auto_provision_users", fallback=True
        )
        oauth_cfg.allow_basic_auth_fallback = config.getboolean(
            "oauth", "allow_basic_auth_fallback", fallback=False
        )
        oauth_cfg.auto_redirect_single_provider = config.getboolean(
            "oauth", "auto_redirect_single_provider", fallback=False
        )

    # Parse OIDC providers: sections like [oauth.oidc.<name>]
    for section in config.sections():
        if section.startswith("oauth.oidc."):
            provider_name = section.split(".", 2)[2]
            provider = OIDCProviderConfig(
                name=provider_name,
                enabled=config.getboolean(section, "enabled", fallback=True),
                display_name=_get_str(config, section, "display_name", "Sign in with SSO"),
                discovery_url=_get_str(config, section, "discovery_url"),
                auth_url=_get_str(config, section, "auth_url"),
                token_url=_get_str(config, section, "token_url"),
                userinfo_url=_get_str(config, section, "userinfo_url"),
                jwks_uri=_get_str(config, section, "jwks_uri"),
                end_session_endpoint=_get_str(config, section, "end_session_endpoint"),
                client_id=_get_str(config, section, "client_id"),
                client_secret=_get_str(config, section, "client_secret"),
                scopes=_get_str(config, section, "scopes", "openid profile email"),
                redirect_uri=_get_str(config, section, "redirect_uri"),
                username_claim=_get_str(config, section, "username_claim", "preferred_username"),
                email_claim=_get_str(config, section, "email_claim", "email"),
                groups_claim=_get_str(config, section, "groups_claim", "groups"),
                name_claim=_get_str(config, section, "name_claim", "name"),
                role_mappings=_get_str(config, section, "role_mappings"),
                admin_groups=_get_str(config, section, "admin_groups"),
                expected_audience=_get_str(config, section, "expected_audience"),
                clock_skew_seconds=config.getint(section, "clock_skew_seconds", fallback=30),
                extra_auth_params=_get_str(config, section, "extra_auth_params"),
            )
            oauth_cfg.oidc_providers[provider_name] = provider

    # Parse SAML providers: sections like [oauth.saml.<name>]
    for section in config.sections():
        if section.startswith("oauth.saml."):
            provider_name = section.split(".", 2)[2]
            provider = SAMLProviderConfig(
                name=provider_name,
                enabled=config.getboolean(section, "enabled", fallback=False),
                display_name=_get_str(config, section, "display_name", "Corporate SSO (SAML)"),
                idp_metadata_url=_get_str(config, section, "idp_metadata_url"),
                idp_metadata_file=_get_str(config, section, "idp_metadata_file"),
                sp_entity_id=_get_str(config, section, "sp_entity_id", "mlflow"),
                sp_acs_url=_get_str(config, section, "sp_acs_url"),
                sp_slo_url=_get_str(config, section, "sp_slo_url"),
                username_attribute=_get_str(config, section, "username_attribute"),
                email_attribute=_get_str(config, section, "email_attribute"),
                groups_attribute=_get_str(config, section, "groups_attribute"),
                role_mappings=_get_str(config, section, "role_mappings"),
                admin_groups=_get_str(config, section, "admin_groups"),
                sp_cert_file=_get_str(config, section, "sp_cert_file"),
                sp_key_file=_get_str(config, section, "sp_key_file"),
                want_assertions_signed=config.getboolean(
                    section, "want_assertions_signed", fallback=True
                ),
            )
            oauth_cfg.saml_providers[provider_name] = provider

    # Parse external authz: [oauth.external_authz]
    if config.has_section("oauth.external_authz"):
        section = "oauth.external_authz"
        oauth_cfg.external_authz = ExternalAuthzConfig(
            enabled=config.getboolean(section, "enabled", fallback=False),
            endpoint=_get_str(config, section, "endpoint"),
            forward_token=config.getboolean(section, "forward_token", fallback=True),
            headers=_get_str(config, section, "headers"),
            allowed_field=_get_str(config, section, "allowed_field", "allowed"),
            permission_field=_get_str(config, section, "permission_field", "permission"),
            admin_field=_get_str(config, section, "admin_field", "is_admin"),
            cache_ttl_seconds=config.getint(section, "cache_ttl_seconds", fallback=300),
            cache_max_size=config.getint(section, "cache_max_size", fallback=10000),
            timeout_seconds=config.getint(section, "timeout_seconds", fallback=5),
            max_retries=config.getint(section, "max_retries", fallback=1),
            retry_backoff_seconds=config.getfloat(section, "retry_backoff_seconds", fallback=0.5),
            on_error=_get_str(config, section, "on_error", "deny"),
        )

    return oauth_cfg
