import textwrap

import pytest

from mlflow.server.auth.oauth.config import (
    OAuthConfig,
    read_oauth_config,
)


@pytest.fixture
def config_file(tmp_path):
    def _write(content):
        path = tmp_path / "oauth.ini"
        path.write_text(textwrap.dedent(content))
        return str(path)

    return _write


def _make_oauth_config():
    from mlflow.server.auth.config import AuthConfig

    auth_config = AuthConfig(
        default_permission="READ",
        database_uri="sqlite:///auth.db",
        admin_username="admin",
        admin_password="password",
        authorization_function="mlflow.server.auth.oauth:authenticate_request_oauth",
        grant_default_workspace_access=False,
        workspace_cache_max_size=10000,
        workspace_cache_ttl_seconds=3600,
    )
    return OAuthConfig(auth_config=auth_config)


def test_read_oauth_config_minimal(config_file, monkeypatch):
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    assert config.auth_config.default_permission == "READ"
    assert config.auth_config.admin_username == "admin"
    assert len(config.oidc_providers) == 0
    assert len(config.saml_providers) == 0


def test_read_oauth_config_oidc_provider_parsing(config_file, monkeypatch):
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth

        [oauth]
        session_lifetime_seconds = 3600
        session_cookie_name = test_session
        encryption_key = abcd1234abcd1234abcd1234abcd1234

        [oauth.oidc.primary]
        enabled = true
        display_name = Test SSO
        discovery_url = https://idp.example.com/.well-known/openid-configuration
        client_id = test-client
        client_secret = test-secret
        scopes = openid profile email groups
        username_claim = preferred_username
        email_claim = email
        groups_claim = groups
        name_claim = name
        role_mappings = readers:READ, editors:EDIT, managers:MANAGE
        admin_groups = admins
        expected_audience = test-client
        clock_skew_seconds = 30
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    assert "primary" in config.oidc_providers
    provider = config.oidc_providers["primary"]
    assert provider.enabled is True
    assert provider.display_name == "Test SSO"
    assert provider.client_id == "test-client"
    assert provider.username_claim == "preferred_username"
    assert provider.groups_claim == "groups"
    assert provider.clock_skew_seconds == 30
    assert config.session_lifetime_seconds == 3600
    assert config.session_cookie_name == "test_session"


def test_read_oauth_config_saml_provider_parsing(config_file, monkeypatch):
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth

        [oauth.saml.corporate]
        enabled = true
        display_name = Corporate SAML
        idp_metadata_url = https://idp.example.com/saml/metadata
        sp_entity_id = mlflow
        username_attribute = urn:oid:0.9.2342.19200300.100.1.1
        email_attribute = urn:oid:0.9.2342.19200300.100.1.3
        groups_attribute = urn:oid:1.3.6.1.4.1.5923.1.5.1.1
        role_mappings = readers:READ, editors:EDIT
        admin_groups = admins
        want_assertions_signed = true
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    assert "corporate" in config.saml_providers
    provider = config.saml_providers["corporate"]
    assert provider.enabled is True
    assert provider.display_name == "Corporate SAML"
    assert provider.sp_entity_id == "mlflow"
    assert provider.want_assertions_signed is True


def test_read_oauth_config_external_authz(config_file, monkeypatch):
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth

        [oauth.external_authz]
        enabled = true
        endpoint = https://authz.example.com/v1/check
        forward_token = true
        cache_ttl_seconds = 600
        cache_max_size = 5000
        timeout_seconds = 10
        max_retries = 2
        retry_backoff_seconds = 1.0
        on_error = fallback_to_default
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    authz = config.external_authz
    assert authz.enabled is True
    assert authz.endpoint == "https://authz.example.com/v1/check"
    assert authz.cache_ttl_seconds == 600
    assert authz.cache_max_size == 5000
    assert authz.timeout_seconds == 10
    assert authz.max_retries == 2
    assert authz.on_error == "fallback_to_default"


def test_read_oauth_config_env_var_substitution(config_file, monkeypatch):
    monkeypatch.setenv("TEST_SECRET_123", "my-secret-value")
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth

        [oauth.oidc.test]
        enabled = true
        display_name = Test
        discovery_url = https://idp.example.com/.well-known/openid-configuration
        client_id = test
        client_secret = ${TEST_SECRET_123}
        scopes = openid
        username_claim = sub
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    assert config.oidc_providers["test"].client_secret == "my-secret-value"


def test_read_oauth_config_get_enabled_providers(config_file, monkeypatch):
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth

        [oauth.oidc.enabled_one]
        enabled = true
        display_name = Enabled
        discovery_url = https://idp.example.com/.well-known/openid-configuration
        client_id = test
        client_secret = secret
        scopes = openid
        username_claim = sub

        [oauth.oidc.disabled_one]
        enabled = false
        display_name = Disabled
        discovery_url = https://idp2.example.com/.well-known/openid-configuration
        client_id = test2
        client_secret = secret2
        scopes = openid
        username_claim = sub
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    providers = config.get_enabled_providers()
    assert len(providers) == 1
    assert providers[0]["name"] == "enabled_one"


def test_read_oauth_config_multiple_providers(config_file, monkeypatch):
    path = config_file("""\
        [mlflow]
        default_permission = READ
        database_uri = sqlite:///auth.db
        admin_username = admin
        admin_password = password1234
        authorization_function = mlflow.server.auth.oauth:authenticate_request_oauth

        [oauth.oidc.google]
        enabled = true
        display_name = Google
        discovery_url = https://accounts.google.com/.well-known/openid-configuration
        client_id = g-client
        client_secret = g-secret
        scopes = openid email
        username_claim = email

        [oauth.oidc.github]
        enabled = true
        display_name = GitHub
        discovery_url = https://github.com/.well-known/openid-configuration
        client_id = gh-client
        client_secret = gh-secret
        scopes = openid
        username_claim = login
    """)
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", path)

    config = read_oauth_config()
    assert len(config.oidc_providers) == 2
    providers = config.get_enabled_providers()
    assert len(providers) == 2


def test_oauth_config_methods_parse_role_mappings():
    config = _make_oauth_config()
    mappings = config.parse_role_mappings("readers:READ, editors:EDIT, managers:MANAGE")
    assert mappings == {"readers": "READ", "editors": "EDIT", "managers": "MANAGE"}


def test_oauth_config_methods_parse_admin_groups():
    config = _make_oauth_config()
    groups = config.parse_admin_groups("admins, super-admins")
    assert groups == ["admins", "super-admins"]


def test_oauth_config_methods_empty_role_mappings():
    config = _make_oauth_config()
    assert config.parse_role_mappings("") == {}


def test_oauth_config_methods_empty_admin_groups():
    config = _make_oauth_config()
    assert config.parse_admin_groups("") == []
