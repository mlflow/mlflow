from __future__ import annotations

from abc import abstractmethod


class WorkspaceIsolatedModel:
    """Mixin for DB models that require workspace-scoped queries.

    Models that participate in workspace isolation must subclass this mixin and
    implement :meth:`workspace_query_filter` to define how queries on the model
    are restricted to a given workspace.

    Note: this class intentionally does not inherit from ``ABC`` because
    SQLAlchemy's ``DeclarativeMeta`` is incompatible with ``ABCMeta``.
    The ``@abstractmethod`` decorator serves as documentation; actual
    enforcement is provided by ``test_all_workspace_isolated_models_implement_filter``.
    """

    @classmethod
    @abstractmethod
    def workspace_query_filter(cls, query, session, workspace):
        """Apply workspace filtering to *query* and return the filtered query.

        Implementations should use one of three strategies depending on the
        model's schema:

        - **Direct column filter** — the model has its own ``workspace`` column:
          ``return query.filter(cls.workspace == workspace)``

        - **Join filter** — the model inherits workspace through a FK parent:
          ``return query.join(Parent, ...).filter(Parent.workspace == workspace)``

        - **Subquery filter** — join is not possible or not efficient:
          ``return query.filter(cls.parent_id.in_(subquery))``
        """
