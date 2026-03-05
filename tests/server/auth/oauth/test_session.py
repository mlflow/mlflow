import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy
from cryptography.exceptions import InvalidTag
from sqlalchemy.orm import sessionmaker

from mlflow.server.auth.oauth.db.models import Base as OAuthBase
from mlflow.server.auth.oauth.session import (
    SessionManager,
    decrypt_token,
    encrypt_token,
    generate_session_id,
)


@dataclass
class FakeOAuthConfig:
    session_lifetime_seconds: int = 3600
    idle_timeout_seconds: int = 1800
    session_refresh_threshold_seconds: int = 300
    session_cookie_name: str = "mlflow_session"
    session_cookie_secure: bool = False
    encryption_key: str = "a" * 64


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
def config():
    return FakeOAuthConfig()


@pytest.fixture
def manager(session_factory, config):
    return SessionManager(session_factory, config)


def test_generate_session_id_length():
    sid = generate_session_id()
    assert len(sid) == 64


def test_generate_session_id_unique():
    ids = {generate_session_id() for _ in range(100)}
    assert len(ids) == 100


def test_token_encryption_round_trip():
    key = "a" * 64
    plaintext = "my-secret-access-token"
    encrypted = encrypt_token(plaintext, key)
    assert encrypted != plaintext
    decrypted = decrypt_token(encrypted, key)
    assert decrypted == plaintext


def test_token_encryption_different_key_fails():
    key1 = "a" * 64
    key2 = "b" * 64
    encrypted = encrypt_token("secret", key1)
    with pytest.raises(InvalidTag):  # noqa: PT011
        decrypt_token(encrypted, key2)


def test_token_encryption_empty_string():
    key = "a" * 64
    encrypted = encrypt_token("", key)
    assert decrypt_token(encrypted, key) == ""


def test_session_manager_create_and_validate(manager):
    now = datetime.now(timezone.utc)
    sid = manager.create_session(
        user_id=1,
        provider="oidc:primary",
        access_token="at123",
        refresh_token="rt456",
        id_token_claims={"username": "jane", "email": "jane@example.com"},
        token_expiry=now + timedelta(hours=1),
        ip_address="10.0.0.1",
        user_agent="pytest",
    )
    assert len(sid) == 64

    info = manager.validate_session(sid)
    assert info is not None
    assert info["user_id"] == 1
    assert info["provider"] == "oidc:primary"
    claims = info["id_token_claims"]
    assert claims["username"] == "jane"


def test_session_manager_validate_expired(session_factory):
    config = FakeOAuthConfig(session_lifetime_seconds=0)
    manager = SessionManager(session_factory, config)

    sid = manager.create_session(
        user_id=1,
        provider="oidc:test",
        access_token="",
        refresh_token="",
        id_token_claims={},
        token_expiry=datetime.now(timezone.utc),
        ip_address="",
        user_agent="",
    )

    time.sleep(0.1)
    info = manager.validate_session(sid)
    assert info is None


def test_session_manager_validate_nonexistent(manager):
    assert manager.validate_session("nonexistent") is None


def test_session_manager_delete(manager):
    sid = manager.create_session(
        user_id=1,
        provider="oidc:test",
        access_token="",
        refresh_token="",
        id_token_claims={},
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        ip_address="",
        user_agent="",
    )

    deleted = manager.delete_session(sid)
    assert deleted is not None
    assert manager.validate_session(sid) is None


def test_session_manager_get_tokens(manager):
    sid = manager.create_session(
        user_id=1,
        provider="oidc:test",
        access_token="access-tok",
        refresh_token="refresh-tok",
        id_token_claims={},
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        ip_address="",
        user_agent="",
    )

    tokens = manager.get_session_tokens(sid)
    assert tokens["access_token"] == "access-tok"
    assert tokens["refresh_token"] == "refresh-tok"


def test_session_manager_update_tokens_keeps_session_id(manager):
    sid = manager.create_session(
        user_id=1,
        provider="oidc:test",
        access_token="old-at",
        refresh_token="old-rt",
        id_token_claims={},
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        ip_address="",
        user_agent="",
    )

    returned_sid = manager.update_session_tokens(
        old_session_id=sid,
        access_token="new-at",
        refresh_token="new-rt",
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=2),
    )

    assert returned_sid == sid
    assert manager.validate_session(sid) is not None

    tokens = manager.get_session_tokens(sid)
    assert tokens["access_token"] == "new-at"
    assert tokens["refresh_token"] == "new-rt"


def test_session_manager_delete_user_sessions(manager):
    for _ in range(3):
        manager.create_session(
            user_id=42,
            provider="oidc:test",
            access_token="",
            refresh_token="",
            id_token_claims={},
            token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            ip_address="",
            user_agent="",
        )

    count = manager.delete_user_sessions(42)
    assert count == 3


def test_session_manager_cleanup_expired(session_factory):
    config = FakeOAuthConfig(session_lifetime_seconds=0)
    manager = SessionManager(session_factory, config)

    manager.create_session(
        user_id=1,
        provider="oidc:test",
        access_token="",
        refresh_token="",
        id_token_claims={},
        token_expiry=datetime.now(timezone.utc),
        ip_address="",
        user_agent="",
    )

    time.sleep(0.1)
    cleaned = manager.cleanup_expired_sessions()
    assert cleaned >= 1


def test_session_manager_should_refresh_token(manager):
    now = datetime.now(timezone.utc)
    # Token expiring soon (within threshold)
    info_soon = {"token_expiry": now + timedelta(seconds=60)}
    assert manager.should_refresh_token(info_soon) is True

    # Token not expiring soon
    info_later = {"token_expiry": now + timedelta(hours=1)}
    assert manager.should_refresh_token(info_later) is False


def test_session_manager_store_and_retrieve_oauth_state(manager):
    manager.store_oauth_state(
        state="state123",
        code_verifier="verifier456",
        nonce="nonce789",
        provider_name="primary",
        redirect_after_login="/experiments",
    )

    data = manager.retrieve_and_delete_oauth_state("state123")
    assert data is not None
    assert data["code_verifier"] == "verifier456"
    assert data["nonce"] == "nonce789"
    assert data["provider_name"] == "primary"
    assert data["redirect_after_login"] == "/experiments"

    # Should be deleted after retrieval (one-time use)
    assert manager.retrieve_and_delete_oauth_state("state123") is None


def test_session_manager_retrieve_nonexistent_state(manager):
    assert manager.retrieve_and_delete_oauth_state("nonexistent") is None


def test_session_manager_cookie_properties(manager):
    from flask import Flask

    app = Flask(__name__)
    sid = "test-session-id-" + "x" * 48

    with app.test_request_context():
        from flask import make_response

        resp = make_response("OK")
        manager.set_session_cookie(resp, sid)

        cookie_header = resp.headers.get("Set-Cookie", "")
        assert "mlflow_session=" in cookie_header
        assert "HttpOnly" in cookie_header
        assert "SameSite=Lax" in cookie_header
        assert "Path=/" in cookie_header
