from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text

from mlflow.server.auth.db.models import Base


class SqlSession(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String(64), nullable=False)
    access_token_enc = Column(Text, nullable=True)
    refresh_token_enc = Column(Text, nullable=True)
    id_token_claims = Column(Text, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    last_accessed_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)

    __table_args__ = (
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_expires_at", "expires_at"),
    )


class SqlOAuthState(Base):
    __tablename__ = "oauth_state"

    state = Column(String(64), primary_key=True)
    code_verifier = Column(String(128), nullable=True)
    nonce = Column(String(64), nullable=True)
    provider_name = Column(String(64), nullable=False)
    redirect_after_login = Column(String(2048), nullable=True)
    created_at = Column(DateTime, nullable=False)


class SqlUserRoleOverride(Base):
    __tablename__ = "user_role_overrides"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    default_permission = Column(String(32), nullable=False)
    idp_groups = Column(Text, nullable=True)
    last_synced_at = Column(DateTime, nullable=False)
