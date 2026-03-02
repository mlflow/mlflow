from __future__ import annotations

from contextlib import contextmanager
from unittest import mock


def mock_get_managed_session_maker(*args, **kwargs):
    @contextmanager
    def _manager():
        session = mock.MagicMock()

        def _mock_query(*q_args, **q_kwargs):
            query = mock.MagicMock()
            query.filter.return_value = query
            query.order_by.return_value = query
            query.exists.return_value = query
            query.scalar.return_value = False
            query.first.return_value = None
            return query

        session.query.side_effect = _mock_query
        yield session

    return _manager
