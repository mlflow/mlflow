from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Column, String, Text

from mlflow.entities.workspace import Workspace
from mlflow.store.db.base_sql_model import Base


class SqlWorkspace(Base):
    __tablename__ = "workspaces"

    # Workspace-aware tables intentionally do not declare SQL-level foreign keys to this model.
    # The WorkspaceProvider abstraction can source workspace metadata from providers outside this
    # database (e.g., Kubernetes namespaces), so referential integrity is enforced at the provider
    # layer rather than through the ORM schema.

    name = Column(String(63), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(
        sa.TIMESTAMP(),
        nullable=False,
        default=datetime.utcnow,
        server_default=sa.func.now(),
    )

    __table_args__ = (sa.PrimaryKeyConstraint("name", name="workspaces_pk"),)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<SqlWorkspace ({self.name})>"

    def to_mlflow_entity(self) -> Workspace:
        return Workspace(name=self.name, description=self.description)
