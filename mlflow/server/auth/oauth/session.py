import base64
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    return secrets.token_hex(32)


def _get_aes_key(encryption_key: str) -> bytes:
    key_bytes = bytes.fromhex(encryption_key)
    if len(key_bytes) != 32:
        raise ValueError("Encryption key must be 64 hex characters (32 bytes)")
    return key_bytes


def encrypt_token(token: str, encryption_key: str) -> str:
    if not token:
        return ""
    key = _get_aes_key(encryption_key)
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    ciphertext = aesgcm.encrypt(nonce, token.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_token(encrypted: str, encryption_key: str) -> str:
    if not encrypted:
        return ""
    key = _get_aes_key(encryption_key)
    aesgcm = AESGCM(key)
    raw = base64.b64decode(encrypted)
    nonce = raw[:12]
    ciphertext = raw[12:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


class SessionManager:
    def __init__(self, session_maker, oauth_config):
        self._session_maker = session_maker
        self._config = oauth_config

    def create_session(
        self,
        user_id: int,
        provider: str,
        access_token: str = "",
        refresh_token: str = "",
        id_token_claims: dict[str, object] | None = None,
        token_expiry: datetime | None = None,
        ip_address: str = "",
        user_agent: str = "",
    ) -> str:
        from mlflow.server.auth.oauth.db.models import SqlSession

        session_id = generate_session_id()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._config.session_lifetime_seconds)
        encryption_key = self._config.encryption_key

        db_session = SqlSession(
            id=session_id,
            user_id=user_id,
            provider=provider,
            access_token_enc=encrypt_token(access_token, encryption_key) if access_token else None,
            refresh_token_enc=(
                encrypt_token(refresh_token, encryption_key) if refresh_token else None
            ),
            id_token_claims=json.dumps(id_token_claims) if id_token_claims else None,
            token_expiry=token_expiry,
            created_at=now,
            last_accessed_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        with self._session_maker() as db:
            db.add(db_session)
            db.commit()

        return session_id

    def validate_session(self, session_id: str) -> dict[str, object] | None:
        from mlflow.server.auth.oauth.db.models import SqlSession

        if not session_id:
            return None

        now = datetime.now(timezone.utc)

        with self._session_maker() as db:
            session = db.query(SqlSession).filter(SqlSession.id == session_id).first()
            if not session:
                return None

            # Check hard expiry
            if session.expires_at.replace(tzinfo=timezone.utc) < now:
                db.delete(session)
                db.commit()
                return None

            # Check idle timeout
            idle_cutoff = now - timedelta(seconds=self._config.idle_timeout_seconds)
            if session.last_accessed_at.replace(tzinfo=timezone.utc) < idle_cutoff:
                db.delete(session)
                db.commit()
                return None

            # Update last accessed
            session.last_accessed_at = now
            db.commit()

            return {
                "session_id": session.id,
                "user_id": session.user_id,
                "provider": session.provider,
                "id_token_claims": (
                    json.loads(session.id_token_claims) if session.id_token_claims else None
                ),
                "token_expiry": session.token_expiry,
                "created_at": session.created_at,
                "expires_at": session.expires_at,
            }

    def get_session_tokens(self, session_id: str) -> dict[str, str] | None:
        from mlflow.server.auth.oauth.db.models import SqlSession

        with self._session_maker() as db:
            session = db.query(SqlSession).filter(SqlSession.id == session_id).first()
            if not session:
                return None

            encryption_key = self._config.encryption_key
            return {
                "access_token": (
                    decrypt_token(session.access_token_enc, encryption_key)
                    if session.access_token_enc
                    else ""
                ),
                "refresh_token": (
                    decrypt_token(session.refresh_token_enc, encryption_key)
                    if session.refresh_token_enc
                    else ""
                ),
            }

    def update_session_tokens(
        self,
        old_session_id: str,
        access_token: str,
        refresh_token: str = "",
        token_expiry: datetime | None = None,
    ) -> str:
        from mlflow.server.auth.oauth.db.models import SqlSession

        encryption_key = self._config.encryption_key

        with self._session_maker() as db:
            session = db.query(SqlSession).filter(SqlSession.id == old_session_id).first()
            if not session:
                return ""

            session.access_token_enc = encrypt_token(access_token, encryption_key)
            if refresh_token:
                session.refresh_token_enc = encrypt_token(refresh_token, encryption_key)
            if token_expiry:
                session.token_expiry = token_expiry
            session.last_accessed_at = datetime.now(timezone.utc)
            db.commit()

        return old_session_id

    def delete_session(self, session_id: str) -> dict[str, str | None] | None:
        from mlflow.server.auth.oauth.db.models import SqlSession

        with self._session_maker() as db:
            session = db.query(SqlSession).filter(SqlSession.id == session_id).first()
            if not session:
                return None
            info = {"provider": session.provider, "id_token_claims": session.id_token_claims}
            db.delete(session)
            db.commit()
            return info

    def delete_user_sessions(self, user_id: int) -> int:
        from mlflow.server.auth.oauth.db.models import SqlSession

        with self._session_maker() as db:
            count = db.query(SqlSession).filter(SqlSession.user_id == user_id).delete()
            db.commit()
            return count

    def cleanup_expired_sessions(self) -> int:
        from mlflow.server.auth.oauth.db.models import SqlOAuthState, SqlSession

        now = datetime.now(timezone.utc)

        with self._session_maker() as db:
            # Clean expired sessions
            session_count = db.query(SqlSession).filter(SqlSession.expires_at < now).delete()

            # Clean expired OAuth state entries (10 minute TTL)
            state_cutoff = now - timedelta(minutes=10)
            state_count = (
                db.query(SqlOAuthState).filter(SqlOAuthState.created_at < state_cutoff).delete()
            )

            db.commit()
            _logger.debug(
                "Cleaned up %d expired sessions and %d expired OAuth state entries",
                session_count,
                state_count,
            )
            return session_count + state_count

    def set_session_cookie(self, response, session_id: str):
        response.set_cookie(
            key=self._config.session_cookie_name,
            value=session_id,
            httponly=True,
            secure=self._config.session_cookie_secure,
            samesite="Lax",
            max_age=self._config.session_lifetime_seconds,
            path="/",
        )

    def clear_session_cookie(self, response):
        response.delete_cookie(
            key=self._config.session_cookie_name,
            path="/",
            samesite="Lax",
        )

    def get_session_id_from_cookie(self, request) -> str:
        return request.cookies.get(self._config.session_cookie_name, "")

    def should_refresh_token(self, session_info: dict[str, object]) -> bool:
        if not session_info or not session_info.get("token_expiry"):
            return False
        token_expiry = session_info["token_expiry"]
        if token_expiry.tzinfo is None:
            token_expiry = token_expiry.replace(tzinfo=timezone.utc)
        threshold = datetime.now(timezone.utc) + timedelta(
            seconds=self._config.session_refresh_threshold_seconds
        )
        return token_expiry <= threshold

    # OAuth state management for PKCE flow
    def store_oauth_state(
        self,
        state: str,
        code_verifier: str,
        nonce: str,
        provider_name: str,
        redirect_after_login: str = "/",
    ):
        from mlflow.server.auth.oauth.db.models import SqlOAuthState

        oauth_state = SqlOAuthState(
            state=state,
            code_verifier=code_verifier,
            nonce=nonce,
            provider_name=provider_name,
            redirect_after_login=redirect_after_login,
            created_at=datetime.now(timezone.utc),
        )

        with self._session_maker() as db:
            db.add(oauth_state)
            db.commit()

    def retrieve_and_delete_oauth_state(self, state: str) -> dict[str, str] | None:
        from mlflow.server.auth.oauth.db.models import SqlOAuthState

        with self._session_maker() as db:
            oauth_state = db.query(SqlOAuthState).filter(SqlOAuthState.state == state).first()
            if not oauth_state:
                return None

            # Check TTL (10 minutes)
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
            if oauth_state.created_at.replace(tzinfo=timezone.utc) < cutoff:
                db.delete(oauth_state)
                db.commit()
                return None

            result = {
                "state": oauth_state.state,
                "code_verifier": oauth_state.code_verifier,
                "nonce": oauth_state.nonce,
                "provider_name": oauth_state.provider_name,
                "redirect_after_login": oauth_state.redirect_after_login or "/",
            }
            db.delete(oauth_state)
            db.commit()
            return result
