import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
import sqlalchemy
from flask import Flask
from sqlalchemy.orm import sessionmaker
from werkzeug.datastructures import Authorization

from mlflow.server.auth.oauth.db.models import Base as OAuthBase
from mlflow.server.auth.oauth.session import SessionManager


@dataclass
class FakeOAuthConfig:
    session_lifetime_seconds: int = 3600
    idle_timeout_seconds: int = 1800
    session_refresh_threshold_seconds: int = 300
    session_cookie_name: str = "mlflow_session"
    session_cookie_secure: bool = False
    encryption_key: str = "a" * 64
    auto_provision_users: bool = True
    allow_basic_auth_fallback: bool = True
    auto_redirect_single_provider: bool = True
    oidc_providers: dict[str, object] = field(default_factory=dict)
    saml_providers: dict[str, object] = field(default_factory=dict)
    external_authz: object | None = None

    def get_enabled_providers(self):
        providers = []
        for name, cfg in self.oidc_providers.items():
            if getattr(cfg, "enabled", True):
                providers.append(
                    {"name": name, "display_name": getattr(cfg, "display_name", ""), "type": "oidc"}
                )
        return providers


@dataclass
class FakeExternalAuthz:
    enabled: bool = False


@dataclass
class FakeOIDCProvider:
    name: str = "keycloak"
    enabled: bool = True
    display_name: str = "Sign in with Keycloak"
    client_id: str = "mlflow"


@pytest.fixture
def db_engine():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(db_engine):
    maker = sessionmaker(bind=db_engine)

    @contextmanager
    def managed():
        s = maker()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return managed


@pytest.fixture
def session_manager(session_factory):
    config = FakeOAuthConfig()
    return SessionManager(session_factory, config)


@pytest.fixture
def oauth_config():
    return FakeOAuthConfig(
        external_authz=FakeExternalAuthz(),
        oidc_providers={"keycloak": FakeOIDCProvider()},
    )


@pytest.fixture
def flask_app():
    app = Flask(__name__)
    app.secret_key = "test-secret"
    return app


def test_enrich_server_info_adds_auth_type_and_providers(flask_app, oauth_config, session_manager):
    import mlflow.server.auth.oauth.app as oauth_app

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        flask_app.test_request_context("/ajax-api/3.0/mlflow/server-info"),
    ):
        resp = flask_app.make_response(json.dumps({"store_type": "SqlStore"}))
        resp.content_type = "application/json"

        oauth_app._enrich_server_info_response(resp)

        data = resp.get_json()
        assert data["auth_type"] == "oauth"
        assert data["store_type"] == "SqlStore"
        assert len(data["auth_providers"]) == 1
        assert data["auth_providers"][0]["name"] == "keycloak"
        assert data["auth_providers"][0]["type"] == "oidc"


def test_enrich_server_info_adds_auth_user_when_authenticated(
    flask_app, oauth_config, session_manager
):
    import mlflow.server.auth.oauth.app as oauth_app

    sid = session_manager.create_session(
        user_id=1,
        provider="oidc:keycloak",
        access_token="at",
        refresh_token="rt",
        id_token_claims={
            "username": "alice",
            "email": "alice@example.com",
            "display_name": "Alice",
        },
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        ip_address="",
        user_agent="",
    )

    mock_user = mock.Mock()
    mock_user.is_admin = True

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        mock.patch("mlflow.server.auth.oauth.app.store") as mock_store,
        flask_app.test_request_context(
            "/ajax-api/3.0/mlflow/server-info",
            headers={"Cookie": f"mlflow_session={sid}"},
        ),
    ):
        from flask import request

        request.username = "alice"
        mock_store.get_user.return_value = mock_user

        resp = flask_app.make_response(json.dumps({"store_type": "SqlStore"}))
        resp.content_type = "application/json"

        oauth_app._enrich_server_info_response(resp)

        data = resp.get_json()
        assert data["auth_user"]["username"] == "alice"
        assert data["auth_user"]["email"] == "alice@example.com"
        assert data["auth_user"]["is_admin"] is True
        mock_store.get_user.assert_called_once_with("alice")


def test_enrich_server_info_no_auth_user_when_not_authenticated(
    flask_app, oauth_config, session_manager
):
    import mlflow.server.auth.oauth.app as oauth_app

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        flask_app.test_request_context("/ajax-api/3.0/mlflow/server-info"),
    ):
        resp = flask_app.make_response(json.dumps({"store_type": "SqlStore"}))
        resp.content_type = "application/json"

        oauth_app._enrich_server_info_response(resp)

        data = resp.get_json()
        assert data["auth_type"] == "oauth"
        assert "auth_user" not in data


def test_enrich_server_info_handles_non_json_gracefully(flask_app, oauth_config, session_manager):
    import mlflow.server.auth.oauth.app as oauth_app

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        flask_app.test_request_context("/ajax-api/3.0/mlflow/server-info"),
    ):
        resp = flask_app.make_response("not json")
        resp.content_type = "text/plain"

        # Should not raise
        oauth_app._enrich_server_info_response(resp)
        assert resp.get_data(as_text=True) == "not json"


def test_logout_returns_json_with_redirect_url(flask_app, oauth_config, session_manager):
    import mlflow.server.auth.oauth.app as oauth_app

    sid = session_manager.create_session(
        user_id=1,
        provider="oidc:keycloak",
        access_token="at",
        refresh_token="rt",
        id_token_claims={"username": "alice"},
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        ip_address="",
        user_agent="",
    )

    mock_endpoints = {
        "end_session_endpoint": (
            "http://keycloak:8080/realms/mlflow/protocol/openid-connect/logout"
        )
    }

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        mock.patch.object(oauth_app, "_external_authz", None),
        mock.patch(
            "mlflow.server.auth.oauth.oidc._get_endpoints",
            return_value=mock_endpoints,
        ) as mock_get_ep,
        flask_app.test_request_context(
            "/auth/logout",
            method="POST",
            headers={"Cookie": f"mlflow_session={sid}"},
        ),
    ):
        response = oauth_app._logout()

        data = response.get_json()
        assert "redirect_url" in data
        assert "keycloak:8080" in data["redirect_url"]
        assert "post_logout_redirect_uri" in data["redirect_url"]
        assert "client_id=mlflow" in data["redirect_url"]
        mock_get_ep.assert_called_once()


def test_logout_without_session_returns_login_url(flask_app, oauth_config, session_manager):
    import mlflow.server.auth.oauth.app as oauth_app

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        mock.patch.object(oauth_app, "_external_authz", None),
        flask_app.test_request_context("/auth/logout", method="POST"),
    ):
        response = oauth_app._logout()

        data = response.get_json()
        assert data["redirect_url"] == "/auth/login"


def test_logout_clears_session_cookie(flask_app, oauth_config, session_manager):
    import mlflow.server.auth.oauth.app as oauth_app

    with (
        mock.patch.object(oauth_app, "_oauth_config", oauth_config),
        mock.patch.object(oauth_app, "_session_manager", session_manager),
        mock.patch.object(oauth_app, "_external_authz", None),
        flask_app.test_request_context("/auth/logout", method="POST"),
    ):
        response = oauth_app._logout()

        cookie_header = response.headers.get("Set-Cookie", "")
        assert "mlflow_session=" in cookie_header
        # Cookie should be expired (cleared)
        assert "Expires=Thu, 01 Jan 1970" in cookie_header or "Max-Age=0" in cookie_header


def test_authenticate_request_shortcut_returns_stored_username(flask_app):
    import mlflow.server.auth as auth_module
    import mlflow.server.auth.oauth.app as oauth_app

    original = auth_module.authenticate_request

    with (
        mock.patch.object(
            auth_module,
            "authenticate_request",
            side_effect=AssertionError("should not call"),
        ),
        flask_app.test_request_context("/api/2.0/mlflow/experiments/search"),
    ):
        oauth_app._install_authenticate_request_shortcut()

        from flask import request

        request.username = "bob"

        result = auth_module.authenticate_request()
        assert isinstance(result, Authorization)
        assert result.username == "bob"

    # Restore
    auth_module.authenticate_request = original


def test_authenticate_request_shortcut_falls_through_without_username(flask_app):
    import mlflow.server.auth as auth_module
    import mlflow.server.auth.oauth.app as oauth_app

    original = auth_module.authenticate_request
    sentinel = Authorization("basic", {"username": "fallback"})

    with (
        mock.patch.object(auth_module, "authenticate_request", return_value=sentinel),
        flask_app.test_request_context("/api/2.0/mlflow/experiments/search"),
    ):
        oauth_app._install_authenticate_request_shortcut()

        result = auth_module.authenticate_request()
        assert result.username == "fallback"

    # Restore
    auth_module.authenticate_request = original
