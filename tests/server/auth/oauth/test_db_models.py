from datetime import datetime, timezone

import sqlalchemy
from sqlalchemy.orm import Session as DbSession

from mlflow.server.auth.oauth.db.models import (
    Base as OAuthBase,
)
from mlflow.server.auth.oauth.db.models import (
    SqlOAuthState,
    SqlSession,
    SqlUserRoleOverride,
)


def test_database_models_create_tables():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)

    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()
    assert "sessions" in tables
    assert "oauth_state" in tables
    assert "user_role_overrides" in tables
    engine.dispose()


def test_database_models_session_crud():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)
    now = datetime.now(timezone.utc)

    with DbSession(engine) as db:
        session = SqlSession(
            id="test-session-id",
            user_id=1,
            provider="oidc:primary",
            access_token_enc="encrypted-at",
            refresh_token_enc="encrypted-rt",
            id_token_claims='{"username": "jane"}',
            token_expiry=now,
            created_at=now,
            last_accessed_at=now,
            expires_at=now,
            ip_address="10.0.0.1",
            user_agent="pytest",
        )
        db.add(session)
        db.commit()

        loaded = db.query(SqlSession).filter_by(id="test-session-id").first()
        assert loaded.user_id == 1
        assert loaded.provider == "oidc:primary"
        assert loaded.ip_address == "10.0.0.1"

    engine.dispose()


def test_database_models_oauth_state_crud():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)
    now = datetime.now(timezone.utc)

    with DbSession(engine) as db:
        state = SqlOAuthState(
            state="state123",
            code_verifier="verifier456",
            nonce="nonce789",
            provider_name="primary",
            redirect_after_login="/experiments",
            created_at=now,
        )
        db.add(state)
        db.commit()

        loaded = db.query(SqlOAuthState).filter_by(state="state123").first()
        assert loaded.code_verifier == "verifier456"
        assert loaded.nonce == "nonce789"
        assert loaded.provider_name == "primary"

    engine.dispose()


def test_database_models_user_role_override_crud():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)
    now = datetime.now(timezone.utc)

    with DbSession(engine) as db:
        override = SqlUserRoleOverride(
            user_id=1,
            default_permission="EDIT",
            idp_groups='["editors", "developers"]',
            last_synced_at=now,
        )
        db.add(override)
        db.commit()

        loaded = db.query(SqlUserRoleOverride).filter_by(user_id=1).first()
        assert loaded.default_permission == "EDIT"
        assert '"editors"' in loaded.idp_groups

    engine.dispose()


def test_database_models_session_indexes_exist():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)

    inspector = sqlalchemy.inspect(engine)
    indexes = inspector.get_indexes("sessions")
    index_names = [idx["name"] for idx in indexes]
    assert "idx_sessions_user_id" in index_names
    assert "idx_sessions_expires_at" in index_names

    engine.dispose()
