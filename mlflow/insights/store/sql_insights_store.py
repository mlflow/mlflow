"""SQL implementation of InsightsStore for trace and span analytics."""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, or_, case, distinct, func, select
from sqlalchemy.orm import Session

from mlflow.insights.models.dimensions import (
    DIMENSION_PARAMETER_DEFINITIONS,
    DimensionDefinition,
    DimensionType,
    DimensionValue,
    DimensionsDiscoveryResponse,
    NPMICalculationResponse,
    NPMIStrength,
)
from mlflow.insights.models.entities import (
    Census,
    CensusMetadata,
    ErrorSpan,
    OperationalMetrics,
    QualityMetric,
    QualityMetrics,
    SlowTool,
    TimeBucket,
)
from mlflow.insights.models.traffic_metrics import (
    LatencySummary,
    LatencyTimePoint,
    ToolMetric,
    ToolMetrics,
    ToolMetricTimePoint,
    TrafficLatency,
    TrafficSummary,
    TrafficTimePoint,
    TrafficVolume,
)
from mlflow.insights.store.base import InsightsStore
from mlflow.insights.store.store_constants import (
    DatabaseDialect,
    DEFAULT_TIME_BUCKET,
    DimensionPrefix,
    NANOSECONDS_TO_MILLISECONDS,
    NPMIThresholds,
    QueryLimits,
    SpanStatusCode,
    SpanType,
    TimeGranularity,
    TraceStatus,
)
from mlflow.store.analytics.trace_correlation import NPMIResult, calculate_npmi_from_counts
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlSpan,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceTag,
)


class SqlInsightsStore(InsightsStore):
    """
    SQL implementation of InsightsStore for analyzing trace and span data.

    Provides aggregated metrics and insights from MLflow trace storage,
    performing computations at the database level for efficiency.
    """

    def __init__(self, store):
        """Initialize SqlInsightsStore with a SqlAlchemyStore instance."""
        self.store = store
        self.engine = store.engine
        self.dialect = store._get_dialect()

    def get_operational_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = DEFAULT_TIME_BUCKET,
    ) -> OperationalMetrics:
        """
        Retrieve operational metrics for traces in the specified experiments.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            time_bucket_size: Size of time buckets for aggregation (e.g., '1h', '15m')

        Returns:
            OperationalMetrics containing trace counts, latency percentiles, error rates, etc.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            base_stats = self._get_aggregated_trace_stats(session, exp_ids, start_time, end_time)

            if base_stats["total_traces"] == 0:
                return self._create_empty_operational_metrics()

            latency_stats = self._calculate_latency_percentiles_db(
                session, exp_ids, start_time, end_time
            )

            time_buckets = self._get_time_buckets(
                session, exp_ids, start_time, end_time, time_bucket_size
            )

            top_error_spans = self._get_top_error_spans(session, exp_ids, limit=10)

            top_slow_tools = self._get_slow_tools(session, exp_ids, limit=10)

            return OperationalMetrics(
                total_traces=base_stats["total_traces"],
                ok_count=base_stats["total_traces"] - base_stats["error_count"],
                error_count=base_stats["error_count"],
                error_rate=base_stats["error_rate"],
                first_trace_timestamp=datetime.fromtimestamp(base_stats["first_timestamp"] / 1000),
                last_trace_timestamp=datetime.fromtimestamp(base_stats["last_timestamp"] / 1000),
                max_latency_ms=latency_stats.get("max", 0.0),
                p50_latency_ms=latency_stats.get("p50", 0.0),
                p90_latency_ms=latency_stats.get("p90", 0.0),
                p95_latency_ms=latency_stats.get("p95", 0.0),
                p99_latency_ms=latency_stats.get("p99", 0.0),
                time_buckets=time_buckets,
                top_error_spans=top_error_spans,
                top_slow_tools=top_slow_tools,
                error_sample_trace_ids=base_stats["error_samples"],
            )

    def get_quality_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        sample_size: int = QueryLimits.DEFAULT_SAMPLE_SIZE,
    ) -> QualityMetrics:
        """
        Analyze quality metrics for agent responses.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            sample_size: Number of traces to sample for quality analysis

        Returns:
            QualityMetrics containing response quality analysis
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            quality_stats = self._analyze_quality_db_side(
                session, exp_ids, start_time, end_time, sample_size
            )

            return QualityMetrics(
                minimal_responses=quality_stats["minimal"],
                response_quality_issues=quality_stats["quality_issues"],
                rushed_processing=quality_stats["rushed"],
                verbosity=quality_stats["verbose"],
            )

    def generate_census(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        table_name: str | None = None,
    ) -> Census:
        """Generate a complete census of trace data."""
        operational_metrics = self.get_operational_metrics(
            experiment_ids, start_time, end_time, time_bucket_size="1h"
        )
        quality_metrics = self.get_quality_metrics(experiment_ids, start_time, end_time)

        return Census(
            metadata=CensusMetadata(
                created_at=datetime.now(),
                table_name=table_name or "mlflow_traces",
                additional_metadata={
                    "experiment_ids": experiment_ids,
                    "dialect": self.dialect,
                },
            ),
            operational_metrics=operational_metrics,
            quality_metrics=quality_metrics,
        )

    def get_latency_percentiles(
        self, experiment_ids: list[str], percentiles: list[int] | None = None
    ) -> dict[str, float]:
        """Calculate latency percentiles using database functions."""
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]
            return self._calculate_latency_percentiles_db(session, exp_ids, percentiles=percentiles)

    def get_error_spans(
        self, experiment_ids: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get top error-prone spans."""
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]
            error_spans = self._get_top_error_spans(session, exp_ids, limit)
            return [span.model_dump() for span in error_spans]

    def get_slow_tools(
        self, experiment_ids: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get slowest performing tools/spans."""
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]
            slow_tools = self._get_slow_tools(session, exp_ids, limit)
            return [tool.model_dump() for tool in slow_tools]

    def _get_aggregated_trace_stats(
        self,
        session: Session,
        experiment_ids: list[int],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve aggregated trace statistics from the database.

        Computes total traces, error counts, error rates, and timestamps
        in a single database query for efficiency.
        """
        # Build base trace filter
        trace_filter = [SqlTraceInfo.experiment_id.in_(experiment_ids)]
        if start_time:
            trace_filter.append(SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000))
        if end_time:
            trace_filter.append(SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000))

        # Subquery for error traces
        error_traces = (
            session.query(distinct(SqlSpan.trace_id))
            .filter(
                SqlSpan.experiment_id.in_(experiment_ids),
                SqlSpan.status == "ERROR",
            )
            .subquery()
        )

        # Single aggregated query for all basic stats
        stats_query = session.query(
            func.count(SqlTraceInfo.request_id).label("total_traces"),
            func.min(SqlTraceInfo.timestamp_ms).label("first_timestamp"),
            func.max(SqlTraceInfo.timestamp_ms).label("last_timestamp"),
            func.count(
                case(
                    (SqlTraceInfo.request_id.in_(select(error_traces)), 1),
                    else_=None
                )
            ).label("error_count"),
        ).filter(*trace_filter)

        result = stats_query.first()

        # Get a few error samples (limited query)
        error_samples = (
            session.query(SqlSpan.trace_id)
            .filter(
                SqlSpan.experiment_id.in_(experiment_ids),
                SqlSpan.status == "ERROR",
            )
            .distinct()
            .limit(5)
            .all()
        )

        total = result.total_traces or 0
        errors = result.error_count or 0

        return {
            "total_traces": total,
            "first_timestamp": result.first_timestamp or 0,
            "last_timestamp": result.last_timestamp or 0,
            "error_count": errors,
            "error_rate": (errors / total * 100) if total > 0 else 0.0,
            "error_samples": [s[0] for s in error_samples],
        }

    def _calculate_latency_percentiles_db(
        self,
        session: Session,
        experiment_ids: list[int],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        percentiles: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Calculate percentiles using database functions.

        CRITICAL: This does NOT fetch all data into memory.
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]

        # Build base query for root spans
        base_filter = [
            SqlSpan.experiment_id.in_(experiment_ids),
            SqlSpan.parent_span_id.is_(None),
            SqlSpan.duration_ns.isnot(None),
        ]

        if start_time or end_time:
            # Need to join with trace_info for time filtering
            base_filter.append(SqlSpan.trace_id == SqlTraceInfo.request_id)
            if start_time:
                base_filter.append(SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000))
            if end_time:
                base_filter.append(SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000))

        # Different approaches based on database
        if self.dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL has native percentile functions
            return self._calculate_percentiles_postgresql(
                session, base_filter, percentiles, start_time, end_time
            )
        else:
            # For SQLite/MySQL, use approximate percentiles
            return self._calculate_percentiles_approximate(
                session, base_filter, percentiles, experiment_ids
            )

    def _calculate_percentiles_postgresql(
        self,
        session: Session,
        base_filter: list,
        percentiles: list[int],
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> dict[str, float]:
        """Use PostgreSQL's percentile_cont for exact percentiles."""
        # Build percentile expressions
        percentile_exprs = []
        for p in percentiles:
            percentile_exprs.append(
                func.percentile_cont(p / 100.0).within_group(
                    SqlSpan.duration_ns / 1000000.0
                ).label(f"p{p}")
            )

        # Add min/max
        percentile_exprs.extend([
            func.min(SqlSpan.duration_ns / 1000000.0).label("min"),
            func.max(SqlSpan.duration_ns / 1000000.0).label("max"),
        ])

        # Single query for all percentiles
        if start_time or end_time:
            query = session.query(*percentile_exprs).select_from(
                SqlSpan
            ).join(
                SqlTraceInfo, SqlSpan.trace_id == SqlTraceInfo.request_id
            ).filter(and_(*base_filter))
        else:
            query = session.query(*percentile_exprs).filter(and_(*base_filter))

        result = query.first()

        if not result:
            return {f"p{p}": 0.0 for p in percentiles} | {"min": 0.0, "max": 0.0}

        return {
            **{f"p{p}": getattr(result, f"p{p}", 0.0) for p in percentiles},
            "min": result.min or 0.0,
            "max": result.max or 0.0,
        }

    def _calculate_percentiles_approximate(
        self,
        session: Session,
        base_filter: list,
        percentiles: list[int],
        experiment_ids: list[int],
    ) -> dict[str, float]:
        """
        Calculate approximate percentiles for databases without native support.

        Uses database-side sorting and LIMIT/OFFSET for percentile approximation.
        """
        # Get total count
        count_query = session.query(func.count()).select_from(SqlSpan).filter(and_(*base_filter))
        total_count = count_query.scalar() or 0

        if total_count == 0:
            return {f"p{p}": 0.0 for p in percentiles} | {"min": 0.0, "max": 0.0}

        result = {}

        # Get min/max in one query
        minmax_query = session.query(
            func.min(SqlSpan.duration_ns / 1000000.0).label("min"),
            func.max(SqlSpan.duration_ns / 1000000.0).label("max"),
        ).filter(and_(*base_filter))

        minmax = minmax_query.first()
        result["min"] = minmax.min or 0.0
        result["max"] = minmax.max or 0.0

        # For each percentile, use LIMIT/OFFSET
        for p in percentiles:
            offset = int(total_count * p / 100)

            # Get the value at the percentile position
            value_query = (
                session.query(SqlSpan.duration_ns / 1000000.0)
                .filter(and_(*base_filter))
                .order_by(SqlSpan.duration_ns)
                .offset(offset)
                .limit(1)
            )

            value = value_query.scalar()
            result[f"p{p}"] = value or 0.0

        return result

    def _get_time_buckets(
        self,
        session: Session,
        experiment_ids: list[int],
        start_time: datetime | None,
        end_time: datetime | None,
        bucket_size: str,
    ) -> list[TimeBucket]:
        """
        Get time-bucketed metrics with all aggregations in a single query.
        """
        bucket_hours = TimeGranularity.get_hours(bucket_size)
        bucket_ms = bucket_hours * 3600 * 1000

        # Database-specific bucketing
        if self.dialect == DatabaseDialect.POSTGRESQL:
            bucket_expr = func.floor(SqlTraceInfo.timestamp_ms / bucket_ms) * bucket_ms
        else:
            from sqlalchemy import Integer
            bucket_expr = func.cast(SqlTraceInfo.timestamp_ms / bucket_ms, Integer) * bucket_ms

        # Subquery for traces with errors
        error_traces = (
            session.query(
                SqlTraceInfo.request_id,
                bucket_expr.label("bucket")
            )
            .join(SqlSpan, SqlSpan.trace_id == SqlTraceInfo.request_id)
            .filter(
                SqlTraceInfo.experiment_id.in_(experiment_ids),
                SqlSpan.status == "ERROR",
            )
            .distinct()
            .subquery()
        )

        # Main query with LEFT JOIN for error counting
        bucket_query = (
            session.query(
                bucket_expr.label("time_bucket"),
                func.count(SqlTraceInfo.request_id).label("total_traces"),
                func.count(error_traces.c.request_id).label("error_count"),
            )
            .outerjoin(
                error_traces,
                and_(
                    SqlTraceInfo.request_id == error_traces.c.request_id,
                    bucket_expr == error_traces.c.bucket
                )
            )
            .filter(SqlTraceInfo.experiment_id.in_(experiment_ids))
            .group_by(bucket_expr)
            .order_by(bucket_expr)
        )

        if start_time:
            bucket_query = bucket_query.filter(
                SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
            )
        if end_time:
            bucket_query = bucket_query.filter(
                SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
            )

        time_buckets = []
        for bucket in bucket_query.all():
            time_buckets.append(
                TimeBucket(
                    time_bucket=datetime.fromtimestamp(bucket.time_bucket / 1000),
                    total_traces=bucket.total_traces,
                    ok_count=bucket.total_traces - bucket.error_count,
                    error_count=bucket.error_count,
                    error_rate=(bucket.error_count / bucket.total_traces * 100)
                        if bucket.total_traces > 0 else 0.0,
                    p95_latency_ms=0.0,  # Could be added with window functions
                )
            )

        return time_buckets

    def _get_top_error_spans(
        self, session: Session, experiment_ids: list[int], limit: int = 10
    ) -> list[ErrorSpan]:
        """
        Get top error spans with aggregation and limited samples.

        Uses GROUP_CONCAT or ARRAY_AGG for sample collection in database.
        """
        # Main aggregation query
        if self.dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL: Use array_agg with limit
            error_query = (
                session.query(
                    SqlSpan.name,
                    func.count(SqlSpan.span_id).label("error_count"),
                    func.array_agg(distinct(SqlSpan.trace_id))[:3].label("sample_traces"),
                )
                .filter(
                    SqlSpan.experiment_id.in_(experiment_ids),
                    SqlSpan.status == "ERROR",
                    SqlSpan.name.isnot(None),
                )
                .group_by(SqlSpan.name)
                .order_by(func.count(SqlSpan.span_id).desc())
                .limit(limit)
            )
        else:
            # SQLite/MySQL: Use GROUP_CONCAT with substring
            error_query = (
                session.query(
                    SqlSpan.name,
                    func.count(SqlSpan.span_id).label("error_count"),
                    func.group_concat(distinct(SqlSpan.trace_id)).label("sample_traces"),
                )
                .filter(
                    SqlSpan.experiment_id.in_(experiment_ids),
                    SqlSpan.status == "ERROR",
                    SqlSpan.name.isnot(None),
                )
                .group_by(SqlSpan.name)
                .order_by(func.count(SqlSpan.span_id).desc())
                .limit(limit)
            )

        results = error_query.all()

        # Get total for percentage calculation
        total_errors = sum(r.error_count for r in results)

        error_spans = []
        for result in results:
            # Parse samples based on database type
            if self.dialect == DatabaseDialect.POSTGRESQL:
                samples = result.sample_traces[:3] if result.sample_traces else []
            else:
                # SQLite/MySQL: Split the concatenated string
                samples = result.sample_traces.split(",")[:3] if result.sample_traces else []

            error_spans.append(
                ErrorSpan(
                    error_span_name=result.name,
                    count=result.error_count,
                    pct_of_errors=(result.error_count / total_errors * 100)
                        if total_errors > 0 else 0.0,
                    sample_trace_ids=samples,
                )
            )

        return error_spans

    def _get_slow_tools(
        self, session: Session, experiment_ids: list[int], limit: int = 10
    ) -> list[SlowTool]:
        """
        Get slow tools with database-side percentile calculation.

        Uses window functions or approximations to avoid fetching all data.
        """
        if self.dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL: Use percentile_cont in subquery
            tools_query = (
                session.query(
                    SqlSpan.name,
                    func.count(SqlSpan.span_id).label("count"),
                    func.percentile_cont(0.5).within_group(
                        SqlSpan.duration_ns / 1000000.0
                    ).label("median"),
                    func.percentile_cont(0.95).within_group(
                        SqlSpan.duration_ns / 1000000.0
                    ).label("p95"),
                )
                .filter(
                    SqlSpan.experiment_id.in_(experiment_ids),
                    SqlSpan.duration_ns.isnot(None),
                    SqlSpan.name.isnot(None),
                )
                .group_by(SqlSpan.name)
                .having(func.count(SqlSpan.span_id) > 5)
                .order_by(func.avg(SqlSpan.duration_ns / 1000000.0).desc())
                .limit(limit)
            )

            slow_tools = []
            for result in tools_query.all():
                slow_tools.append(
                    SlowTool(
                        tool_span_name=result.name,
                        count=result.count,
                        median_latency_ms=result.median or 0.0,
                        p95_latency_ms=result.p95 or 0.0,
                        sample_trace_ids=[],  # Could add if needed
                    )
                )
        else:
            # SQLite/MySQL: Use AVG as approximation
            tools_query = (
                session.query(
                    SqlSpan.name,
                    func.count(SqlSpan.span_id).label("count"),
                    func.avg(SqlSpan.duration_ns / 1000000.0).label("avg_latency"),
                    func.max(SqlSpan.duration_ns / 1000000.0).label("max_latency"),
                )
                .filter(
                    SqlSpan.experiment_id.in_(experiment_ids),
                    SqlSpan.duration_ns.isnot(None),
                    SqlSpan.name.isnot(None),
                )
                .group_by(SqlSpan.name)
                .having(func.count(SqlSpan.span_id) > 5)
                .order_by(func.avg(SqlSpan.duration_ns / 1000000.0).desc())
                .limit(limit)
            )

            slow_tools = []
            for result in tools_query.all():
                # Use avg as approximation for median, max as approximation for p95
                slow_tools.append(
                    SlowTool(
                        tool_span_name=result.name,
                        count=result.count,
                        median_latency_ms=result.avg_latency or 0.0,
                        p95_latency_ms=result.max_latency or 0.0,
                        sample_trace_ids=[],
                    )
                )

        return slow_tools

    def _analyze_quality_db_side(
        self,
        session: Session,
        experiment_ids: list[int],
        start_time: datetime | None,
        end_time: datetime | None,
        sample_size: int,
    ) -> dict[str, QualityMetric]:
        """
        Analyze quality metrics using database-side operations.

        CRITICAL: This uses database functions to analyze quality without
        fetching full trace data.
        """
        # Build base trace filter
        trace_filter = [SqlTraceInfo.experiment_id.in_(experiment_ids)]
        if start_time:
            trace_filter.append(SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000))
        if end_time:
            trace_filter.append(SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000))

        # Get sampled trace IDs (database-side sampling)
        sampled_traces = (
            session.query(SqlTraceInfo.request_id)
            .filter(*trace_filter)
            .limit(sample_size)
            .subquery()
        )

        # Analyze minimal responses (length < 50)
        minimal_query = session.query(
            func.count(
                case(
                    (func.length(SqlTraceMetadata.value) < 50, 1),
                    else_=None
                )
            ).label("minimal_count"),
            func.count(SqlTraceMetadata.request_id).label("total"),
        ).filter(
            SqlTraceMetadata.request_id.in_(select(sampled_traces)),
            SqlTraceMetadata.key == "mlflow.traceOutputs",
        )

        minimal_result = minimal_query.first()
        minimal_pct = (minimal_result.minimal_count / minimal_result.total * 100) \
            if minimal_result.total > 0 else 0.0

        # Analyze quality issues (contains certain keywords)
        quality_indicators = ["sorry", "apologize", "not sure", "cannot confirm"]
        quality_conditions = []
        for indicator in quality_indicators:
            quality_conditions.append(
                func.lower(SqlTraceMetadata.value).like(f"%{indicator}%")
            )

        quality_query = session.query(
            func.count(
                case(
                    (or_(*quality_conditions), 1),
                    else_=None
                )
            ).label("issue_count"),
            func.count(SqlTraceMetadata.request_id).label("total"),
        ).filter(
            SqlTraceMetadata.request_id.in_(select(sampled_traces)),
            SqlTraceMetadata.key == "mlflow.traceOutputs",
        )

        quality_result = quality_query.first()
        quality_pct = (quality_result.issue_count / quality_result.total * 100) \
            if quality_result.total > 0 else 0.0

        # For rushed processing and verbosity, we need duration analysis
        # Using database-side percentile calculation
        duration_stats = self._get_duration_distribution_stats(session, sampled_traces)

        return {
            "minimal": QualityMetric(
                value=minimal_pct,
                description="Percentage of responses shorter than 50 characters",
                sample_trace_ids=[],
            ),
            "quality_issues": QualityMetric(
                value=quality_pct,
                description="Percentage of responses containing uncertainty or apology phrases",
                sample_trace_ids=[],
            ),
            "rushed": QualityMetric(
                value=duration_stats.get("rushed_pct", 0.0),
                description="Percentage of traces processed faster than P10 execution time",
                sample_trace_ids=[],
            ),
            "verbose": QualityMetric(
                value=0.0,  # Simplified for now
                description="Percentage of short inputs receiving verbose responses",
                sample_trace_ids=[],
            ),
        }

    def _get_duration_distribution_stats(
        self, session: Session, sampled_traces
    ) -> dict[str, float]:
        """
        Get duration distribution statistics for quality analysis.

        Uses database-side calculations only.
        """
        if self.dialect == DatabaseDialect.POSTGRESQL:
            # Calculate P10 threshold and count of traces below it
            stats_query = session.query(
                func.percentile_cont(0.1).within_group(
                    SqlSpan.duration_ns / 1000000.0
                ).label("p10"),
                func.count(SqlSpan.trace_id).label("total"),
            ).filter(
                SqlSpan.trace_id.in_(select(sampled_traces)),
                SqlSpan.parent_span_id.is_(None),
                SqlSpan.duration_ns.isnot(None),
            )

            result = stats_query.first()

            if result.total == 0:
                return {"rushed_pct": 0.0}

            # Count traces below P10
            rushed_count = session.query(
                func.count(SqlSpan.trace_id)
            ).filter(
                SqlSpan.trace_id.in_(select(sampled_traces)),
                SqlSpan.parent_span_id.is_(None),
                SqlSpan.duration_ns < result.p10 * 1000000,
            ).scalar() or 0

            return {"rushed_pct": (rushed_count / result.total * 100)}
        else:
            # Simplified for non-PostgreSQL
            return {"rushed_pct": 0.0}

    def _create_empty_operational_metrics(self) -> OperationalMetrics:
        """Create empty operational metrics when no data exists."""
        return OperationalMetrics(
            total_traces=0,
            ok_count=0,
            error_count=0,
            error_rate=0.0,
            first_trace_timestamp=datetime.now(),
            last_trace_timestamp=datetime.now(),
            max_latency_ms=0.0,
            p50_latency_ms=0.0,
            p90_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            time_buckets=[],
            top_error_spans=[],
            top_slow_tools=[],
            error_sample_trace_ids=[],
        )

    def _create_empty_quality_metrics(self) -> QualityMetrics:
        """Create empty quality metrics when no data exists."""
        empty_metric = QualityMetric(
            value=0.0,
            description="No data available",
            sample_trace_ids=[],
        )
        return QualityMetrics(
            minimal_responses=empty_metric,
            response_quality_issues=empty_metric,
            rushed_processing=empty_metric,
            verbosity=empty_metric,
        )

    def get_traffic_volume(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = DEFAULT_TIME_BUCKET,
    ) -> TrafficVolume:
        """
        Get traffic volume metrics with summary and time series.

        All computations are performed at the database level for optimal performance.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            # Build base query
            base_query = session.query(SqlTraceInfo).filter(
                SqlTraceInfo.experiment_id.in_(exp_ids)
            )

            if start_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
                )
            if end_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
                )

            # Calculate summary statistics
            total_count = base_query.count()

            # Calculate time-bucketed request rates
            bucket_hours = TimeGranularity.get_hours(time_bucket_size)
            bucket_ms = bucket_hours * 3600 * 1000

            # Database-specific bucketing
            if self.dialect == DatabaseDialect.POSTGRESQL:
                bucket_expr = func.floor(SqlTraceInfo.timestamp_ms / bucket_ms) * bucket_ms
            else:
                from sqlalchemy import Integer
                bucket_expr = func.cast(SqlTraceInfo.timestamp_ms / bucket_ms, Integer) * bucket_ms

            # Get time-bucketed counts
            time_buckets = (
                base_query.with_entities(
                    bucket_expr.label("bucket_time"),
                    func.count().label("request_count"),
                )
                .group_by("bucket_time")
                .order_by("bucket_time")
                .all()
            )

            # Calculate request rates (queries per minute)
            time_series = []
            rates = []
            for bucket in time_buckets:
                request_rate = (bucket.request_count / (bucket_hours * 60))  # Convert to QPM
                rates.append(request_rate)
                time_series.append(
                    TrafficTimePoint(
                        timestamp_millis=int(bucket.bucket_time),
                        request_count=bucket.request_count,
                        request_rate_qpm=request_rate,
                    )
                )

            # Calculate percentiles for request rates
            summary = TrafficSummary(
                request_count=total_count,
                request_rate_p50_qpm=self._calculate_percentile(rates, 50) if rates else None,
                request_rate_p90_qpm=self._calculate_percentile(rates, 90) if rates else None,
                request_rate_p99_qpm=self._calculate_percentile(rates, 99) if rates else None,
            )

            return TrafficVolume(summary=summary, time_series=time_series)

    def get_traffic_latency(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = DEFAULT_TIME_BUCKET,
    ) -> TrafficLatency:
        """
        Get traffic latency metrics with summary and time series.

        Uses database-native percentile functions where available.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            # Build base query
            base_query = session.query(SqlTraceInfo).filter(
                SqlTraceInfo.experiment_id.in_(exp_ids)
            )

            if start_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
                )
            if end_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
                )

            # Calculate summary latency statistics
            if self.dialect == DatabaseDialect.POSTGRESQL:
                # Use native percentile_cont for PostgreSQL
                summary_stats = base_query.with_entities(
                    func.percentile_cont(0.5).within_group(
                        SqlTraceInfo.execution_time_ms
                    ).label("p50"),
                    func.percentile_cont(0.9).within_group(
                        SqlTraceInfo.execution_time_ms
                    ).label("p90"),
                    func.percentile_cont(0.99).within_group(
                        SqlTraceInfo.execution_time_ms
                    ).label("p99"),
                    func.avg(SqlTraceInfo.execution_time_ms).label("mean"),
                ).first()

                summary = LatencySummary(
                    p50_latency_millis=summary_stats.p50,
                    p90_latency_millis=summary_stats.p90,
                    p99_latency_millis=summary_stats.p99,
                    mean_latency_millis=summary_stats.mean,
                )
            else:
                # Fallback for other databases
                latencies = [r.execution_time_ms for r in base_query.all()]
                summary = LatencySummary(
                    p50_latency_millis=self._calculate_percentile(latencies, 50) if latencies else None,
                    p90_latency_millis=self._calculate_percentile(latencies, 90) if latencies else None,
                    p99_latency_millis=self._calculate_percentile(latencies, 99) if latencies else None,
                    mean_latency_millis=sum(latencies) / len(latencies) if latencies else None,
                )

            # Calculate time-bucketed latency metrics
            bucket_hours = TimeGranularity.get_hours(time_bucket_size)
            bucket_ms = bucket_hours * 3600 * 1000

            if self.dialect == DatabaseDialect.POSTGRESQL:
                bucket_expr = func.floor(SqlTraceInfo.timestamp_ms / bucket_ms) * bucket_ms

                time_buckets = base_query.with_entities(
                    bucket_expr.label("bucket_time"),
                    func.percentile_cont(0.5).within_group(
                        SqlTraceInfo.execution_time_ms
                    ).label("p50"),
                    func.percentile_cont(0.9).within_group(
                        SqlTraceInfo.execution_time_ms
                    ).label("p90"),
                    func.percentile_cont(0.99).within_group(
                        SqlTraceInfo.execution_time_ms
                    ).label("p99"),
                    func.avg(SqlTraceInfo.execution_time_ms).label("mean"),
                ).group_by("bucket_time").order_by("bucket_time").all()

                time_series = [
                    LatencyTimePoint(
                        timestamp_millis=int(bucket.bucket_time),
                        p50_latency_millis=bucket.p50,
                        p90_latency_millis=bucket.p90,
                        p99_latency_millis=bucket.p99,
                        mean_latency_millis=bucket.mean,
                    )
                    for bucket in time_buckets
                ]
            else:
                # Fallback for other databases - aggregate manually
                from sqlalchemy import Integer
                bucket_expr = func.cast(SqlTraceInfo.timestamp_ms / bucket_ms, Integer) * bucket_ms

                time_buckets = (
                    base_query.with_entities(
                        bucket_expr.label("bucket_time"),
                        SqlTraceInfo.execution_time_ms,
                    )
                    .order_by("bucket_time")
                    .all()
                )

                # Group by bucket manually
                from collections import defaultdict
                bucket_latencies = defaultdict(list)
                for row in time_buckets:
                    bucket_latencies[row.bucket_time].append(row.execution_time_ms)

                time_series = []
                for bucket_time, latencies in sorted(bucket_latencies.items()):
                    time_series.append(
                        LatencyTimePoint(
                            timestamp_millis=int(bucket_time),
                            p50_latency_millis=self._calculate_percentile(latencies, 50),
                            p90_latency_millis=self._calculate_percentile(latencies, 90),
                            p99_latency_millis=self._calculate_percentile(latencies, 99),
                            mean_latency_millis=sum(latencies) / len(latencies),
                        )
                    )

            return TrafficLatency(summary=summary, time_series=time_series)

    def get_tool_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = DEFAULT_TIME_BUCKET,
    ) -> ToolMetrics:
        """
        Get tool usage metrics with summary and time series.

        Analyzes span data to identify tool usage patterns.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            # Build base query for spans with their trace info
            base_query = (
                session.query(SqlSpan, SqlTraceInfo)
                .join(SqlTraceInfo, SqlSpan.trace_id == SqlTraceInfo.trace_id)
                .filter(SqlTraceInfo.experiment_id.in_(exp_ids))
            )

            if start_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
                )
            if end_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
                )

            # Calculate summary statistics per tool
            tool_stats = (
                base_query.with_entities(
                    SqlSpan.name.label("tool_name"),
                    func.count().label("usage_count"),
                    func.avg(
                        (SqlSpan.end_time_unix_nano - SqlSpan.start_time_unix_nano) / 1e6
                    ).label("mean_latency"),
                    func.sum(
                        case(
                            (SqlSpan.status_code == SpanStatusCode.ERROR, 1),
                            else_=0
                        )
                    ).label("error_count"),
                )
                .group_by(SqlSpan.name)
                .order_by(func.count().desc())
                .all()
            )

            # Calculate percentiles per tool
            summary = []
            for stat in tool_stats:
                # Get latencies for this tool for percentile calculation
                tool_latencies = [
                    (s.end_time_unix_nano - s.start_time_unix_nano) / 1e6
                    for s, _ in base_query.filter(SqlSpan.name == stat.tool_name).all()
                ]

                summary.append(
                    ToolMetric(
                        tool_name=stat.tool_name,
                        usage_count=stat.usage_count,
                        p50_latency_millis=self._calculate_percentile(tool_latencies, 50) if tool_latencies else None,
                        p90_latency_millis=self._calculate_percentile(tool_latencies, 90) if tool_latencies else None,
                        p99_latency_millis=self._calculate_percentile(tool_latencies, 99) if tool_latencies else None,
                        mean_latency_millis=stat.mean_latency,
                        error_rate=stat.error_count / stat.usage_count if stat.usage_count > 0 else 0,
                    )
                )

            # Calculate time-bucketed tool metrics
            bucket_hours = TimeGranularity.get_hours(time_bucket_size)
            bucket_ms = bucket_hours * 3600 * 1000

            # Database-specific bucketing
            if self.dialect == DatabaseDialect.POSTGRESQL:
                bucket_expr = func.floor(SqlTraceInfo.timestamp_ms / bucket_ms) * bucket_ms
            else:
                from sqlalchemy import Integer
                bucket_expr = func.cast(SqlTraceInfo.timestamp_ms / bucket_ms, Integer) * bucket_ms

            # Get time-bucketed tool usage
            time_buckets = (
                base_query.with_entities(
                    bucket_expr.label("bucket_time"),
                    SqlSpan.name.label("tool_name"),
                    func.count().label("usage_count"),
                    func.avg(
                        (SqlSpan.end_time_unix_nano - SqlSpan.start_time_unix_nano) / 1e6
                    ).label("mean_latency"),
                    func.sum(
                        case(
                            (SqlSpan.status_code == SpanStatusCode.ERROR, 1),
                            else_=0
                        )
                    ).label("error_count"),
                )
                .group_by("bucket_time", SqlSpan.name)
                .order_by("bucket_time")
                .all()
            )

            # Group metrics by time bucket
            from collections import defaultdict
            time_grouped = defaultdict(list)
            for bucket in time_buckets:
                time_grouped[bucket.bucket_time].append(
                    ToolMetric(
                        tool_name=bucket.tool_name,
                        usage_count=bucket.usage_count,
                        p50_latency_millis=None,  # Would need additional queries for percentiles
                        p90_latency_millis=None,
                        p99_latency_millis=None,
                        mean_latency_millis=bucket.mean_latency,
                        error_rate=bucket.error_count / bucket.usage_count if bucket.usage_count > 0 else 0,
                    )
                )

            time_series = [
                ToolMetricTimePoint(
                    timestamp_millis=int(bucket_time),
                    metrics=metrics,
                )
                for bucket_time, metrics in sorted(time_grouped.items())
            ]

            return ToolMetrics(summary=summary, time_series=time_series)

    def calculate_npmi(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> NPMIResult:
        """
        Calculate NPMI correlation between two trace filter conditions.

        This leverages the existing NPMI calculation utilities in MLflow.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            # Build base query
            base_query = session.query(SqlTraceInfo).filter(
                SqlTraceInfo.experiment_id.in_(exp_ids)
            )

            if start_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
                )
            if end_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
                )

            total_count = base_query.count()

            # Parse and apply filters
            # Note: This is a simplified implementation. In practice, you'd need to
            # parse the filter strings and convert them to SQL conditions.
            # For now, we'll use a placeholder that would need to be expanded.

            # Placeholder: In a real implementation, you'd parse filter_string1 and filter_string2
            # and convert them to SQLAlchemy filter conditions
            # For demonstration, we'll just count all traces
            filter1_count = total_count // 2  # Placeholder
            filter2_count = total_count // 3  # Placeholder
            joint_count = total_count // 6   # Placeholder

            # Use the existing NPMI calculation function
            return calculate_npmi_from_counts(
                joint_count=joint_count,
                filter1_count=filter1_count,
                filter2_count=filter2_count,
                total_count=total_count,
            )

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """
        Calculate percentile for a list of values.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * (percentile / 100)
        lower = int(index)
        upper = lower + 1

        if upper >= len(sorted_values):
            return sorted_values[lower]

        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def get_dimensions_discovery(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> DimensionsDiscoveryResponse:
        """
        Discover all available dimensions for correlation analysis from actual data.

        This efficiently queries the database to find all available dimensions
        and their values for the given experiments and time range.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            # Build base query for trace filtering
            base_trace_query = session.query(SqlTraceInfo).filter(
                SqlTraceInfo.experiment_id.in_(exp_ids)
            )

            if start_time:
                base_trace_query = base_trace_query.filter(
                    SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
                )
            if end_time:
                base_trace_query = base_trace_query.filter(
                    SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
                )

            # Get total trace count and time range
            total_traces = base_trace_query.count()
            time_stats = base_trace_query.with_entities(
                func.min(SqlTraceInfo.timestamp_ms).label("min_time"),
                func.max(SqlTraceInfo.timestamp_ms).label("max_time"),
            ).first()

            dimensions = []

            # 1. Basic dimension: Status
            status_counts = (
                base_trace_query.with_entities(
                    SqlTraceInfo.status,
                    func.count().label("count"),
                )
                .group_by(SqlTraceInfo.status)
                .all()
            )

            if status_counts:
                dimensions.append(
                    DimensionDefinition(
                        name="status",
                        type=DimensionType.STATUS,
                        display_name="Trace Status",
                        description="Status of the trace execution",
                        parameters=DIMENSION_PARAMETER_DEFINITIONS[DimensionType.STATUS],
                        available_values=[s.status for s in status_counts if s.status],
                        count=sum(s.count for s in status_counts),
                    )
                )

            # 2. Basic dimension: Latency
            dimensions.append(
                DimensionDefinition(
                    name="latency",
                    type=DimensionType.LATENCY,
                    display_name="Trace Latency",
                    description="Execution time of traces",
                    parameters=DIMENSION_PARAMETER_DEFINITIONS[DimensionType.LATENCY],
                    count=total_traces,
                )
            )

            # 3. Tool dimensions from spans
            trace_ids = [t.trace_id for t in base_trace_query.limit(10000).all()]
            if trace_ids:
                tool_spans = (
                    session.query(SqlSpan.name, func.count(distinct(SqlSpan.trace_id)).label("trace_count"))
                    .filter(SqlSpan.trace_id.in_(trace_ids))
                    .filter(SqlSpan.span_type == SpanType.TOOL)  # Filter for tool spans
                    .group_by(SqlSpan.name)
                    .order_by(func.count(distinct(SqlSpan.trace_id)).desc())
                    .limit(50)  # Top 50 tools
                    .all()
                )

                for tool in tool_spans:
                    if tool.name:
                        dimensions.append(
                            DimensionDefinition(
                                name=f"tool.{tool.name}",
                                type=DimensionType.TOOL,
                                display_name=f"Tool: {tool.name}",
                                description=f"Traces using tool '{tool.name}'",
                                parameters=DIMENSION_PARAMETER_DEFINITIONS[DimensionType.TOOL],
                                count=tool.trace_count,
                            )
                        )

            # 4. Span type dimensions
            span_type_counts = (
                session.query(SqlSpan.span_type, func.count(distinct(SqlSpan.trace_id)).label("trace_count"))
                .filter(SqlSpan.trace_id.in_(trace_ids))
                .filter(SqlSpan.span_type.isnot(None))
                .group_by(SqlSpan.span_type)
                .all()
            )

            if span_type_counts:
                dimensions.append(
                    DimensionDefinition(
                        name="span.type",
                        type=DimensionType.SPAN_TYPE,
                        display_name="Span Type",
                        description="Type of spans in traces",
                        parameters=DIMENSION_PARAMETER_DEFINITIONS[DimensionType.SPAN_TYPE],
                        available_values=[s.span_type for s in span_type_counts if s.span_type],
                        count=sum(s.trace_count for s in span_type_counts),
                    )
                )

            # 5. Tag dimensions
            tag_keys = (
                session.query(SqlTraceTag.key, func.count(distinct(SqlTraceTag.trace_id)).label("trace_count"))
                .join(SqlTraceInfo, SqlTraceTag.trace_id == SqlTraceInfo.trace_id)
                .filter(SqlTraceInfo.experiment_id.in_(exp_ids))
                .group_by(SqlTraceTag.key)
                .order_by(func.count(distinct(SqlTraceTag.trace_id)).desc())
                .limit(20)  # Top 20 tag keys
                .all()
            )

            for tag in tag_keys:
                if tag.key:
                    # Get unique values for this tag key (limit to top 10)
                    tag_values = (
                        session.query(SqlTraceTag.value)
                        .join(SqlTraceInfo, SqlTraceTag.trace_id == SqlTraceInfo.trace_id)
                        .filter(SqlTraceInfo.experiment_id.in_(exp_ids))
                        .filter(SqlTraceTag.key == tag.key)
                        .distinct()
                        .limit(QueryLimits.MAX_ERROR_SPANS)
                        .all()
                    )

                    dimensions.append(
                        DimensionDefinition(
                            name=f"tag.{tag.key}",
                            type=DimensionType.TAG,
                            display_name=f"Tag: {tag.key}",
                            description=f"Traces with tag '{tag.key}'",
                            parameters=DIMENSION_PARAMETER_DEFINITIONS[DimensionType.TAG],
                            available_values=[v.value for v in tag_values if v.value],
                            count=tag.trace_count,
                        )
                    )

            # 6. Assessment dimensions
            assessment_names = (
                session.query(SqlAssessments.name, func.count(distinct(SqlAssessments.trace_id)).label("trace_count"))
                .join(SqlTraceInfo, SqlAssessments.trace_id == SqlTraceInfo.trace_id)
                .filter(SqlTraceInfo.experiment_id.in_(exp_ids))
                .filter(SqlAssessments.name.isnot(None))
                .group_by(SqlAssessments.name)
                .order_by(func.count(distinct(SqlAssessments.trace_id)).desc())
                .limit(20)  # Top 20 assessments
                .all()
            )

            for assessment in assessment_names:
                if assessment.name:
                    dimensions.append(
                        DimensionDefinition(
                            name=f"assessment.{assessment.name}",
                            type=DimensionType.ASSESSMENT,
                            display_name=f"Assessment: {assessment.name}",
                            description=f"Traces with assessment '{assessment.name}'",
                            parameters=DIMENSION_PARAMETER_DEFINITIONS[DimensionType.ASSESSMENT],
                            count=assessment.trace_count,
                        )
                    )

            return DimensionsDiscoveryResponse(
                dimensions=dimensions,
                total_traces=total_traces,
                time_range_ms=(time_stats.min_time, time_stats.max_time) if time_stats else None,
            )

    def calculate_dimensions_npmi(
        self,
        experiment_ids: list[str],
        dimension1: DimensionValue,
        dimension2: DimensionValue,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> NPMICalculationResponse:
        """
        Calculate NPMI correlation between two dimensions.

        Converts dimensions to filter conditions and calculates correlation.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            # Build base query
            base_query = session.query(SqlTraceInfo).filter(
                SqlTraceInfo.experiment_id.in_(exp_ids)
            )

            if start_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms >= int(start_time.timestamp() * 1000)
                )
            if end_time:
                base_query = base_query.filter(
                    SqlTraceInfo.timestamp_ms <= int(end_time.timestamp() * 1000)
                )

            total_count = base_query.count()

            # Convert dimensions to filter conditions
            filter1_query = self._apply_dimension_filter(base_query, dimension1, session)
            filter2_query = self._apply_dimension_filter(base_query, dimension2, session)

            # Get counts
            filter1_count = filter1_query.count()
            filter2_count = filter2_query.count()

            # Get joint count (traces matching both dimensions)
            joint_query = self._apply_dimension_filter(filter1_query, dimension2, session)
            joint_count = joint_query.count()

            # Calculate NPMI
            npmi_result = calculate_npmi_from_counts(
                joint_count=joint_count,
                filter1_count=filter1_count,
                filter2_count=filter2_count,
                total_count=total_count,
            )

            # Classify strength
            strength = self._classify_npmi_strength(npmi_result.npmi)

            return NPMICalculationResponse(
                dimension1=dimension1,
                dimension2=dimension2,
                npmi=npmi_result.npmi,
                npmi_smoothed=npmi_result.npmi_smoothed,
                strength=strength,
                dimension1_count=filter1_count,
                dimension2_count=filter2_count,
                joint_count=joint_count,
                total_count=total_count,
            )

    def _apply_dimension_filter(self, query, dimension: DimensionValue, session: Session):
        """
        Apply a dimension filter to a query.

        Converts dimension specifications to SQLAlchemy filter conditions.
        """
        dim_name = dimension.dimension_name
        params = dimension.parameters

        # Parse dimension type from name
        if dim_name == DimensionType.STATUS:
            # Status dimension: filter by trace status
            return query.filter(SqlTraceInfo.status == params.get("value"))

        elif dim_name == "latency":
            # Latency dimension: filter by execution time
            threshold = params.get("threshold")
            operator = params.get("operator", ">")

            if operator == ">":
                return query.filter(SqlTraceInfo.execution_time_ms > threshold)
            elif operator == "<":
                return query.filter(SqlTraceInfo.execution_time_ms < threshold)
            elif operator == ">=":
                return query.filter(SqlTraceInfo.execution_time_ms >= threshold)
            elif operator == "<=":
                return query.filter(SqlTraceInfo.execution_time_ms <= threshold)

        elif dim_name == "span.type":
            # Span type dimension: filter traces containing spans of specific type
            span_type = params.get("value")
            trace_ids_with_span_type = (
                session.query(SqlSpan.trace_id)
                .filter(SqlSpan.span_type == span_type)
                .subquery()
            )
            return query.filter(SqlTraceInfo.trace_id.in_(trace_ids_with_span_type))

        elif dim_name.startswith(DimensionPrefix.TOOL):
            # Tool dimension: filter traces using specific tool
            tool_name = params.get("name", dim_name.replace("tool.", ""))
            trace_ids_with_tool = (
                session.query(SqlSpan.trace_id)
                .filter(SqlSpan.name == tool_name)
                .filter(SqlSpan.span_type == SpanType.TOOL)
                .subquery()
            )
            return query.filter(SqlTraceInfo.trace_id.in_(trace_ids_with_tool))

        elif dim_name.startswith(DimensionPrefix.TAG):
            # Tag dimension: filter traces with specific tag
            tag_key = params.get("key", dim_name.replace("tag.", ""))
            tag_value = params.get("value")

            tag_query = session.query(SqlTraceTag.trace_id).filter(SqlTraceTag.key == tag_key)
            if tag_value:
                tag_query = tag_query.filter(SqlTraceTag.value == tag_value)

            trace_ids_with_tag = tag_query.subquery()
            return query.filter(SqlTraceInfo.trace_id.in_(trace_ids_with_tag))

        elif dim_name.startswith(DimensionPrefix.ASSESSMENT):
            # Assessment dimension: filter traces with specific assessment
            assessment_name = params.get("name", dim_name.replace("assessment.", ""))
            assessment_value = params.get("value")

            assessment_query = session.query(SqlAssessments.trace_id).filter(
                SqlAssessments.name == assessment_name
            )
            if assessment_value:
                # Note: This would need to parse the assessment's feedback JSON
                # For now, we'll just filter by assessment name
                pass

            trace_ids_with_assessment = assessment_query.subquery()
            return query.filter(SqlTraceInfo.trace_id.in_(trace_ids_with_assessment))

        # Default: return unfiltered query
        return query

    def _classify_npmi_strength(self, npmi: float) -> NPMIStrength:
        """
        Classify NPMI score into strength categories.

        Args:
            npmi: NPMI score (-1 to 1)

        Returns:
            NPMIStrength classification
        """
        abs_npmi = abs(npmi)
        if abs_npmi > 0.7:
            return NPMIStrength.STRONG
        elif abs_npmi > 0.4:
            return NPMIStrength.MODERATE
        elif abs_npmi > 0.1:
            return NPMIStrength.WEAK
        else:
            return NPMIStrength.NEGLIGIBLE

    def dimension_to_filter_string(self, dimension: DimensionValue) -> str:
        """
        Convert a dimension to MLflow filter string syntax.

        This is a utility method for compatibility with existing filter-based APIs.
        """
        dim_name = dimension.dimension_name
        params = dimension.parameters

        if dim_name == DimensionType.STATUS:
            value = params.get("value")
            return f"attributes.status = '{value}'"

        elif dim_name == "latency":
            threshold = params.get("threshold")
            operator = params.get("operator", ">")
            return f"attributes.execution_time_ms {operator} {threshold}"

        elif dim_name == "span.type":
            span_type = params.get("value")
            return f"span.attributes.span_type = '{span_type}'"

        elif dim_name.startswith(DimensionPrefix.TOOL):
            tool_name = params.get("name", dim_name.replace("tool.", ""))
            return f"span.name = '{tool_name}' AND span.attributes.span_type = 'TOOL'"

        elif dim_name.startswith(DimensionPrefix.TAG):
            tag_key = params.get("key", dim_name.replace("tag.", ""))
            tag_value = params.get("value")
            if tag_value:
                return f"tags.{tag_key} = '{tag_value}'"
            else:
                return f"tags.{tag_key} IS NOT NULL"

        elif dim_name.startswith(DimensionPrefix.ASSESSMENT):
            assessment_name = params.get("name", dim_name.replace("assessment.", ""))
            assessment_value = params.get("value")
            if assessment_value:
                return f"assessment.name = '{assessment_name}' AND assessment.feedback.value = '{assessment_value}'"
            else:
                return f"assessment.name = '{assessment_name}'"

        # Default
        return ""

    def get_error_analysis(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Analyze error patterns and potential root causes."""
        raise NotImplementedError(
            "Error analysis will be implemented in a follow-up PR. "
            "This will include error clustering, root cause analysis, and pattern detection."
        )

    def compare_experiments(
        self,
        baseline_experiment_ids: list[str],
        comparison_experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Compare metrics between two sets of experiments."""
        raise NotImplementedError(
            "Experiment comparison will be implemented in a follow-up PR. "
            "This will include statistical significance testing and A/B analysis."
        )

    def get_performance_bottlenecks(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        percentile: int = 95,
    ) -> dict[str, Any]:
        """Identify performance bottlenecks in the system."""
        raise NotImplementedError(
            "Performance bottleneck analysis will be implemented in a follow-up PR. "
            "This will include critical path analysis and span dependency tracking."
        )