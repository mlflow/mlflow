"""
SQLAlchemy mixin for Gateway Usage Tracking storage operations.

This mixin provides methods for logging gateway invocations and provider calls,
as well as querying usage metrics for visualization and analysis.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Integer, and_, func
from sqlalchemy.orm import joinedload

from mlflow.entities.gateway_usage import (
    GatewayInvocation,
    GatewayUsageMetrics,
    InvocationStatus,
    ProviderCallInput,
    ProviderCallStatus,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlGatewayInvocation,
    SqlGatewayProviderCall,
)
from mlflow.utils.time import get_current_time_millis


class SqlAlchemyGatewayUsageStoreMixin:
    """Mixin class providing SQLAlchemy Gateway Usage Tracking implementations.

    This mixin adds usage tracking functionality to SQLAlchemy-based tracking stores,
    enabling logging and querying of gateway invocations and provider calls.

    Requires the base class to provide:
    - ManagedSessionMaker: Context manager for database sessions
    """

    def log_gateway_invocation(
        self,
        endpoint_id: str,
        endpoint_type: str,
        status: InvocationStatus,
        provider_calls: list[ProviderCallInput] | None = None,
        total_latency_ms: int = 0,
        username: str | None = None,
        error_message: str | None = None,
    ) -> GatewayInvocation:
        """
        Log a gateway invocation with its associated provider calls.

        Args:
            endpoint_id: ID of the gateway endpoint that was called.
            endpoint_type: Type of the endpoint (e.g., "llm/v1/chat").
            status: Overall status of the invocation.
            provider_calls: List of ProviderCallInput entities.
            total_latency_ms: Total time taken for the invocation.
            username: Identity of the caller.
            error_message: Error message if the invocation failed.

        Returns:
            The created GatewayInvocation entity.
        """
        with self.ManagedSessionMaker() as session:
            invocation_id = f"i-{uuid.uuid4().hex[:32]}"
            current_time = get_current_time_millis()

            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            total_cost = 0.0

            if provider_calls:
                for pc in provider_calls:
                    total_prompt_tokens += pc.prompt_tokens
                    total_completion_tokens += pc.completion_tokens
                    total_tokens += pc.total_tokens
                    total_cost += pc.total_cost

            sql_invocation = SqlGatewayInvocation(
                invocation_id=invocation_id,
                endpoint_id=endpoint_id,
                endpoint_type=endpoint_type,
                status=status.value,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                total_cost=total_cost,
                total_latency_ms=total_latency_ms,
                created_at=current_time,
                username=username,
                error_message=error_message,
            )
            session.add(sql_invocation)

            if provider_calls:
                for pc in provider_calls:
                    provider_call_id = f"pc-{uuid.uuid4().hex[:31]}"
                    sql_provider_call = SqlGatewayProviderCall(
                        provider_call_id=provider_call_id,
                        invocation_id=invocation_id,
                        provider=pc.provider,
                        model_name=pc.model_name,
                        attempt_number=pc.attempt_number,
                        status=pc.status.value
                        if isinstance(pc.status, ProviderCallStatus)
                        else pc.status,
                        error_message=pc.error_message,
                        prompt_tokens=pc.prompt_tokens,
                        completion_tokens=pc.completion_tokens,
                        total_tokens=pc.total_tokens,
                        prompt_cost=pc.prompt_cost,
                        completion_cost=pc.completion_cost,
                        total_cost=pc.total_cost,
                        latency_ms=pc.latency_ms,
                        created_at=current_time,
                    )
                    session.add(sql_provider_call)

            session.flush()
            return sql_invocation.to_mlflow_entity()

    def get_gateway_invocation(self, invocation_id: str) -> GatewayInvocation | None:
        """
        Get a gateway invocation by ID with its provider calls.

        Args:
            invocation_id: The invocation ID to look up.

        Returns:
            The GatewayInvocation entity or None if not found.
        """
        with self.ManagedSessionMaker() as session:
            sql_invocation = (
                session.query(SqlGatewayInvocation)
                .options(joinedload(SqlGatewayInvocation.provider_calls))
                .filter(SqlGatewayInvocation.invocation_id == invocation_id)
                .first()
            )
            return sql_invocation.to_mlflow_entity() if sql_invocation else None

    def list_gateway_invocations(
        self,
        endpoint_id: str | None = None,
        status: InvocationStatus | None = None,
        username: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        max_results: int = 100,
        page_token: str | None = None,
    ) -> tuple[list[GatewayInvocation], str | None]:
        """
        List gateway invocations with optional filtering.

        Args:
            endpoint_id: Filter by endpoint ID.
            status: Filter by invocation status.
            username: Filter by username.
            start_time: Filter by start time (milliseconds, inclusive).
            end_time: Filter by end time (milliseconds, exclusive).
            max_results: Maximum number of results to return.
            page_token: Token for pagination (invocation_id to start after).

        Returns:
            Tuple of (list of GatewayInvocation entities, next page token or None).
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlGatewayInvocation).options(
                joinedload(SqlGatewayInvocation.provider_calls)
            )

            filters = []
            if endpoint_id:
                filters.append(SqlGatewayInvocation.endpoint_id == endpoint_id)
            if status:
                filters.append(SqlGatewayInvocation.status == status.value)
            if username:
                filters.append(SqlGatewayInvocation.username == username)
            if start_time is not None:
                filters.append(SqlGatewayInvocation.created_at >= start_time)
            if end_time is not None:
                filters.append(SqlGatewayInvocation.created_at < end_time)
            if page_token:
                filters.append(SqlGatewayInvocation.invocation_id > page_token)

            if filters:
                query = query.filter(and_(*filters))

            query = query.order_by(SqlGatewayInvocation.invocation_id)
            query = query.limit(max_results + 1)

            results = query.all()

            next_page_token = None
            if len(results) > max_results:
                results = results[:max_results]
                next_page_token = results[-1].invocation_id

            return [r.to_mlflow_entity() for r in results], next_page_token

    def get_gateway_usage_metrics(
        self,
        endpoint_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        bucket_size: int = 86400,  # Default: 1 day in seconds
    ) -> list[GatewayUsageMetrics]:
        """
        Get aggregated usage metrics for gateway endpoints.

        Args:
            endpoint_id: Filter by endpoint ID (None for all endpoints).
            start_time: Start time in milliseconds (inclusive).
            end_time: End time in milliseconds (exclusive).
            bucket_size: Time bucket size in seconds (e.g., 3600 for hourly, 86400 for daily).

        Returns:
            List of GatewayUsageMetrics entities.
        """
        with self.ManagedSessionMaker() as session:
            bucket_ms = bucket_size * 1000  # Convert seconds to milliseconds

            # Use integer division (floor) to bucket timestamps correctly
            # Without casting, SQLAlchemy may perform floating-point division
            time_bucket_expr = (
                func.cast(SqlGatewayInvocation.created_at / bucket_ms, Integer)
                * bucket_ms
            )

            query = session.query(
                SqlGatewayInvocation.endpoint_id,
                time_bucket_expr.label("time_bucket"),
                func.count().label("total_invocations"),
                func.sum(
                    func.cast(
                        SqlGatewayInvocation.status == InvocationStatus.SUCCESS.value,
                        Integer,
                    )
                ).label("successful_invocations"),
                func.sum(
                    func.cast(
                        SqlGatewayInvocation.status == InvocationStatus.ERROR.value,
                        Integer,
                    )
                ).label("failed_invocations"),
                func.sum(SqlGatewayInvocation.total_prompt_tokens).label(
                    "total_prompt_tokens"
                ),
                func.sum(SqlGatewayInvocation.total_completion_tokens).label(
                    "total_completion_tokens"
                ),
                func.sum(SqlGatewayInvocation.total_tokens).label("total_tokens"),
                func.sum(SqlGatewayInvocation.total_cost).label("total_cost"),
                func.avg(SqlGatewayInvocation.total_latency_ms).label("avg_latency_ms"),
            )

            filters = []
            if endpoint_id:
                filters.append(SqlGatewayInvocation.endpoint_id == endpoint_id)
            if start_time is not None:
                filters.append(SqlGatewayInvocation.created_at >= start_time)
            if end_time is not None:
                filters.append(SqlGatewayInvocation.created_at < end_time)

            if filters:
                query = query.filter(and_(*filters))

            query = query.group_by(
                SqlGatewayInvocation.endpoint_id,
                time_bucket_expr,
            )
            query = query.order_by(
                SqlGatewayInvocation.endpoint_id,
                time_bucket_expr,
            )

            results = query.all()

            return [
                GatewayUsageMetrics(
                    endpoint_id=r.endpoint_id,
                    time_bucket=int(r.time_bucket),
                    bucket_size=bucket_size,
                    total_invocations=r.total_invocations or 0,
                    successful_invocations=r.successful_invocations or 0,
                    failed_invocations=r.failed_invocations or 0,
                    total_prompt_tokens=r.total_prompt_tokens or 0,
                    total_completion_tokens=r.total_completion_tokens or 0,
                    total_tokens=r.total_tokens or 0,
                    total_cost=float(r.total_cost or 0.0),
                    avg_latency_ms=float(r.avg_latency_ms or 0.0),
                )
                for r in results
            ]

    def get_gateway_token_usage_by_endpoint(
        self,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict]:
        """
        Get token usage aggregated by endpoint.

        Args:
            start_time: Start time in milliseconds (inclusive).
            end_time: End time in milliseconds (exclusive).

        Returns:
            List of dicts with endpoint_id, total_prompt_tokens,
            total_completion_tokens, total_tokens, total_cost.
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(
                SqlGatewayInvocation.endpoint_id,
                func.sum(SqlGatewayInvocation.total_prompt_tokens).label(
                    "total_prompt_tokens"
                ),
                func.sum(SqlGatewayInvocation.total_completion_tokens).label(
                    "total_completion_tokens"
                ),
                func.sum(SqlGatewayInvocation.total_tokens).label("total_tokens"),
                func.sum(SqlGatewayInvocation.total_cost).label("total_cost"),
                func.count().label("invocation_count"),
            )

            filters = []
            if start_time is not None:
                filters.append(SqlGatewayInvocation.created_at >= start_time)
            if end_time is not None:
                filters.append(SqlGatewayInvocation.created_at < end_time)

            if filters:
                query = query.filter(and_(*filters))

            query = query.group_by(SqlGatewayInvocation.endpoint_id)
            query = query.order_by(func.sum(SqlGatewayInvocation.total_tokens).desc())

            results = query.all()

            return [
                {
                    "endpoint_id": r.endpoint_id,
                    "total_prompt_tokens": r.total_prompt_tokens or 0,
                    "total_completion_tokens": r.total_completion_tokens or 0,
                    "total_tokens": r.total_tokens or 0,
                    "total_cost": float(r.total_cost or 0.0),
                    "invocation_count": r.invocation_count or 0,
                }
                for r in results
            ]

    def get_gateway_error_rate(
        self,
        endpoint_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        bucket_size: int = 86400,  # Default: 1 day in seconds
    ) -> list[dict]:
        """
        Get error rates by endpoint and time bucket.

        Args:
            endpoint_id: Filter by endpoint ID (None for all endpoints).
            start_time: Start time in milliseconds (inclusive).
            end_time: End time in milliseconds (exclusive).
            bucket_size: Time bucket size in seconds (e.g., 3600 for hourly, 86400 for daily).

        Returns:
            List of dicts with endpoint_id, time_bucket, total_invocations,
            successful_invocations, failed_invocations, success_rate, error_rate.
        """
        with self.ManagedSessionMaker() as session:
            bucket_ms = bucket_size * 1000  # Convert seconds to milliseconds

            # Use integer division (floor) to bucket timestamps correctly
            # Without casting, SQLAlchemy may perform floating-point division
            time_bucket_expr = (
                func.cast(SqlGatewayInvocation.created_at / bucket_ms, Integer)
                * bucket_ms
            )

            query = session.query(
                SqlGatewayInvocation.endpoint_id,
                time_bucket_expr.label("time_bucket"),
                func.count().label("total_invocations"),
                func.sum(
                    func.cast(
                        SqlGatewayInvocation.status == InvocationStatus.SUCCESS.value,
                        Integer,
                    )
                ).label("successful_invocations"),
                func.sum(
                    func.cast(
                        SqlGatewayInvocation.status == InvocationStatus.ERROR.value,
                        Integer,
                    )
                ).label("failed_invocations"),
            )

            filters = []
            if endpoint_id:
                filters.append(SqlGatewayInvocation.endpoint_id == endpoint_id)
            if start_time is not None:
                filters.append(SqlGatewayInvocation.created_at >= start_time)
            if end_time is not None:
                filters.append(SqlGatewayInvocation.created_at < end_time)

            if filters:
                query = query.filter(and_(*filters))

            query = query.group_by(
                SqlGatewayInvocation.endpoint_id,
                time_bucket_expr,
            )
            query = query.order_by(
                SqlGatewayInvocation.endpoint_id,
                time_bucket_expr,
            )

            results = query.all()

            return [
                {
                    "endpoint_id": r.endpoint_id,
                    "time_bucket": int(r.time_bucket),
                    "total_invocations": r.total_invocations or 0,
                    "successful_invocations": r.successful_invocations or 0,
                    "failed_invocations": r.failed_invocations or 0,
                    "success_rate": (
                        (r.successful_invocations or 0) / r.total_invocations * 100
                        if r.total_invocations
                        else 0
                    ),
                    "error_rate": (
                        (r.failed_invocations or 0) / r.total_invocations * 100
                        if r.total_invocations
                        else 0
                    ),
                }
                for r in results
            ]
