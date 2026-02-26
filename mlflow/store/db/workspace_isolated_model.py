from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Query, Session


class WorkspaceIsolatedModel:
    """Mixin for DB models that require workspace-scoped queries.

    Any model that inherits from this must implement ``workspace_query_filter``,
    which defines how to apply workspace filtering to a query on this model.
    """

    @classmethod
    @abstractmethod
    def workspace_query_filter(cls, query: Query, session: Session, workspace: str) -> Query: ...
