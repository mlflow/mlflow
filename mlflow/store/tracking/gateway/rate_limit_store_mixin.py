"""
SQLAlchemy mixin for Gateway Rate Limit storage operations.

This mixin provides methods for CRUD operations on rate limit configurations
for gateway endpoints.
"""

from __future__ import annotations

import uuid

from sqlalchemy import and_

from mlflow.entities.gateway_rate_limit import (
    GatewayRateLimitConfig,
    GatewayRateLimitInput,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.dbmodels.models import SqlGatewayRateLimitConfig
from mlflow.utils.time import get_current_time_millis


class SqlAlchemyGatewayRateLimitStoreMixin:
    """Mixin class providing SQLAlchemy Gateway Rate Limit CRUD implementations.

    This mixin adds rate limit configuration functionality to SQLAlchemy-based
    tracking stores, enabling management of per-endpoint and per-user rate limits.

    Requires the base class to provide:
    - ManagedSessionMaker: Context manager for database sessions
    """

    def create_gateway_rate_limit(
        self,
        endpoint_id: str,
        queries_per_minute: int,
        username: str | None = None,
        created_by: str | None = None,
    ) -> GatewayRateLimitConfig:
        """
        Create a new rate limit configuration.

        Args:
            endpoint_id: ID of the gateway endpoint.
            queries_per_minute: Maximum queries allowed per minute.
            username: Username for per-user limits (None for default endpoint limit).
            created_by: User who created the configuration.

        Returns:
            The created GatewayRateLimitConfig entity.

        Raises:
            MlflowException: If a rate limit already exists for this endpoint/user combo.
        """
        if queries_per_minute <= 0:
            raise MlflowException(
                "queries_per_minute must be a positive integer",
                INVALID_PARAMETER_VALUE,
            )

        with self.ManagedSessionMaker() as session:
            # Check if rate limit already exists
            existing = (
                session.query(SqlGatewayRateLimitConfig)
                .filter(
                    and_(
                        SqlGatewayRateLimitConfig.endpoint_id == endpoint_id,
                        SqlGatewayRateLimitConfig.username == username
                        if username
                        else SqlGatewayRateLimitConfig.username.is_(None),
                    )
                )
                .first()
            )

            if existing:
                user_desc = f"user '{username}'" if username else "default"
                raise MlflowException(
                    f"Rate limit already exists for endpoint '{endpoint_id}' ({user_desc})",
                    RESOURCE_ALREADY_EXISTS,
                )

            rate_limit_id = f"rl-{uuid.uuid4().hex[:32]}"
            current_time = get_current_time_millis()

            sql_rate_limit = SqlGatewayRateLimitConfig(
                rate_limit_id=rate_limit_id,
                endpoint_id=endpoint_id,
                queries_per_minute=queries_per_minute,
                username=username,
                created_at=current_time,
                updated_at=current_time,
                created_by=created_by,
                updated_by=created_by,
            )
            session.add(sql_rate_limit)
            session.flush()
            return sql_rate_limit.to_mlflow_entity()

    def get_gateway_rate_limit(self, rate_limit_id: str) -> GatewayRateLimitConfig | None:
        """
        Get a rate limit configuration by ID.

        Args:
            rate_limit_id: The rate limit ID to look up.

        Returns:
            The GatewayRateLimitConfig entity or None if not found.
        """
        with self.ManagedSessionMaker() as session:
            sql_rate_limit = (
                session.query(SqlGatewayRateLimitConfig)
                .filter(SqlGatewayRateLimitConfig.rate_limit_id == rate_limit_id)
                .first()
            )
            return sql_rate_limit.to_mlflow_entity() if sql_rate_limit else None

    def get_gateway_rate_limit_for_user(
        self,
        endpoint_id: str,
        username: str | None = None,
    ) -> GatewayRateLimitConfig | None:
        """
        Get the effective rate limit for a user on an endpoint.

        First checks for a user-specific rate limit, then falls back to the
        default endpoint rate limit.

        Args:
            endpoint_id: ID of the gateway endpoint.
            username: Username to check (None to get default limit).

        Returns:
            The effective GatewayRateLimitConfig or None if no limit is set.
        """
        with self.ManagedSessionMaker() as session:
            # First try to get user-specific rate limit
            if username:
                user_limit = (
                    session.query(SqlGatewayRateLimitConfig)
                    .filter(
                        and_(
                            SqlGatewayRateLimitConfig.endpoint_id == endpoint_id,
                            SqlGatewayRateLimitConfig.username == username,
                        )
                    )
                    .first()
                )
                if user_limit:
                    return user_limit.to_mlflow_entity()

            # Fall back to default endpoint limit
            default_limit = (
                session.query(SqlGatewayRateLimitConfig)
                .filter(
                    and_(
                        SqlGatewayRateLimitConfig.endpoint_id == endpoint_id,
                        SqlGatewayRateLimitConfig.username.is_(None),
                    )
                )
                .first()
            )
            return default_limit.to_mlflow_entity() if default_limit else None

    def list_gateway_rate_limits(
        self,
        endpoint_id: str | None = None,
        username: str | None = None,
        include_defaults_only: bool = False,
    ) -> list[GatewayRateLimitConfig]:
        """
        List rate limit configurations with optional filtering.

        Args:
            endpoint_id: Filter by endpoint ID.
            username: Filter by username.
            include_defaults_only: If True, only return default (non-user) limits.

        Returns:
            List of GatewayRateLimitConfig entities.
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlGatewayRateLimitConfig)

            filters = []
            if endpoint_id:
                filters.append(SqlGatewayRateLimitConfig.endpoint_id == endpoint_id)
            if username:
                filters.append(SqlGatewayRateLimitConfig.username == username)
            if include_defaults_only:
                filters.append(SqlGatewayRateLimitConfig.username.is_(None))

            if filters:
                query = query.filter(and_(*filters))

            query = query.order_by(
                SqlGatewayRateLimitConfig.endpoint_id,
                SqlGatewayRateLimitConfig.username,
            )

            results = query.all()
            return [r.to_mlflow_entity() for r in results]

    def update_gateway_rate_limit(
        self,
        rate_limit_id: str,
        queries_per_minute: int,
        updated_by: str | None = None,
    ) -> GatewayRateLimitConfig:
        """
        Update a rate limit configuration.

        Args:
            rate_limit_id: ID of the rate limit to update.
            queries_per_minute: New maximum queries per minute.
            updated_by: User who updated the configuration.

        Returns:
            The updated GatewayRateLimitConfig entity.

        Raises:
            MlflowException: If the rate limit does not exist.
        """
        if queries_per_minute <= 0:
            raise MlflowException(
                "queries_per_minute must be a positive integer",
                INVALID_PARAMETER_VALUE,
            )

        with self.ManagedSessionMaker() as session:
            sql_rate_limit = (
                session.query(SqlGatewayRateLimitConfig)
                .filter(SqlGatewayRateLimitConfig.rate_limit_id == rate_limit_id)
                .first()
            )

            if not sql_rate_limit:
                raise MlflowException(
                    f"Rate limit with ID '{rate_limit_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )

            sql_rate_limit.queries_per_minute = queries_per_minute
            sql_rate_limit.updated_at = get_current_time_millis()
            sql_rate_limit.updated_by = updated_by

            session.flush()
            return sql_rate_limit.to_mlflow_entity()

    def delete_gateway_rate_limit(self, rate_limit_id: str) -> None:
        """
        Delete a rate limit configuration.

        Args:
            rate_limit_id: ID of the rate limit to delete.

        Raises:
            MlflowException: If the rate limit does not exist.
        """
        with self.ManagedSessionMaker() as session:
            sql_rate_limit = (
                session.query(SqlGatewayRateLimitConfig)
                .filter(SqlGatewayRateLimitConfig.rate_limit_id == rate_limit_id)
                .first()
            )

            if not sql_rate_limit:
                raise MlflowException(
                    f"Rate limit with ID '{rate_limit_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )

            session.delete(sql_rate_limit)
