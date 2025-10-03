"""SQL implementation of InsightsStore for trace and span analytics."""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, case, distinct, func, or_, select
from sqlalchemy.orm import Session

from mlflow.insights.models.entities import (
    Census,
    CensusMetadata,
    ErrorSpan,
    OperationalMetrics,
    QualityMetric,
    QualityMetrics,
    SlowTool,
)
from mlflow.insights.store.base import InsightsStore
from mlflow.store.tracking.dbmodels.models import (
    SqlSpan,
    SqlTraceInfo,
    SqlTraceMetadata,
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

    def generate_census(
        self,
        experiment_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        table_name: str | None = None,
    ) -> Census:
        """
        Generate a comprehensive census of trace data.

        Analyzes traces to provide statistical distributions and patterns
        including operational metrics, quality assessments, and dimensional analysis.

        Args:
            experiment_id: The experiment ID to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            table_name: Optional table name for metadata

        Returns:
            Census object containing statistics and distributions
        """
        from mlflow.store.tracking.dbmodels.models import SqlTraceInfo

        # Convert single experiment_id to list for internal methods
        # Note: SQL store uses integer IDs internally
        experiment_ids = [experiment_id]

        operational_metrics = self.get_operational_metrics(experiment_ids, start_time, end_time)
        quality_metrics = self.get_quality_metrics(experiment_ids, start_time, end_time)

        # Use the actual table name from the model if not provided
        if table_name is None:
            table_name = SqlTraceInfo.__tablename__

        return Census(
            metadata=CensusMetadata(
                created_at=datetime.now(),
                table_name=table_name,
                additional_metadata={
                    "experiment_id": experiment_id,
                    "dialect": self.dialect,
                },
            ),
            operational_metrics=operational_metrics,
            quality_metrics=quality_metrics,
        )

    def get_operational_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> OperationalMetrics:
        """
        Retrieve operational metrics for traces in the specified experiments.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            OperationalMetrics containing trace counts, latency percentiles, error rates,
            time buckets (max 10), and top error/slow spans.
        """
        with self.store.ManagedSessionMaker() as session:
            exp_ids = [int(e) for e in experiment_ids]

            base_stats = self._get_aggregated_trace_stats(session, exp_ids, start_time, end_time)

            if base_stats["total_traces"] == 0:
                return self._create_empty_operational_metrics()

            latency_stats = self._calculate_latency_percentiles_db(
                session, exp_ids, start_time, end_time
            )

            time_buckets = []  # TODO: Implement smart time buckets that highlight anomalies

            top_error_spans = self._get_top_error_spans(session, exp_ids, limit=5)

            top_slow_tools = self._get_slow_tools(session, exp_ids, limit=5)

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
            )

    def get_quality_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        sample_size: int = 100,
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
                case((SqlTraceInfo.request_id.in_(select(error_traces)), 1), else_=None)
            ).label("error_count"),
        ).filter(*trace_filter)

        result = stats_query.first()

        total = result.total_traces or 0
        errors = result.error_count or 0

        return {
            "total_traces": total,
            "first_timestamp": result.first_timestamp or 0,
            "last_timestamp": result.last_timestamp or 0,
            "error_count": errors,
            "error_rate": (errors / total * 100) if total > 0 else 0.0,
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
        if self.dialect == "postgresql":
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
        base_filter: list[Any],
        percentiles: list[int],
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> dict[str, float]:
        """Use PostgreSQL's percentile_cont for exact percentiles."""
        # Build percentile expressions
        percentile_exprs = []
        for p in percentiles:
            percentile_exprs.append(
                func.percentile_cont(p / 100.0)
                .within_group(SqlSpan.duration_ns / 1000000.0)
                .label(f"p{p}")
            )

        # Add min/max
        percentile_exprs.extend(
            [
                func.min(SqlSpan.duration_ns / 1000000.0).label("min"),
                func.max(SqlSpan.duration_ns / 1000000.0).label("max"),
            ]
        )

        # Single query for all percentiles
        if start_time or end_time:
            query = (
                session.query(*percentile_exprs)
                .select_from(SqlSpan)
                .join(SqlTraceInfo, SqlSpan.trace_id == SqlTraceInfo.request_id)
                .filter(and_(*base_filter))
            )
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
        base_filter: list[Any],
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

    def _get_top_error_spans(
        self, session: Session, experiment_ids: list[int], limit: int = 5
    ) -> list[ErrorSpan]:
        """
        Get top error spans with aggregation and limited samples.

        Uses GROUP_CONCAT or ARRAY_AGG for sample collection in database.
        """
        # Main aggregation query
        if self.dialect == "postgresql":
            # PostgreSQL: Use array_agg with limit
            error_query = (
                session.query(
                    SqlSpan.name,
                    func.count(SqlSpan.span_id).label("error_count"),
                    func.array_agg(distinct(SqlSpan.trace_id))[:10].label("sample_traces"),
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
            if self.dialect == "postgresql":
                samples = result.sample_traces[:10] if result.sample_traces else []
            else:
                # SQLite/MySQL: Split the concatenated string
                samples = result.sample_traces.split(",")[:10] if result.sample_traces else []

            error_spans.append(
                ErrorSpan(
                    error_span_name=result.name,
                    count=result.error_count,
                    pct_of_errors=(result.error_count / total_errors * 100)
                    if total_errors > 0
                    else 0.0,
                    sample_trace_ids=samples,
                )
            )

        return error_spans

    def _get_slow_tools(
        self, session: Session, experiment_ids: list[int], limit: int = 5
    ) -> list[SlowTool]:
        """
        Get slow tools with database-side percentile calculation.

        Uses window functions or approximations to avoid fetching all data.
        """
        if self.dialect == "postgresql":
            # PostgreSQL: Use percentile_cont in subquery
            tools_query = (
                session.query(
                    SqlSpan.name,
                    func.count(SqlSpan.span_id).label("count"),
                    func.percentile_cont(0.5)
                    .within_group(SqlSpan.duration_ns / 1000000.0)
                    .label("median"),
                    func.percentile_cont(0.95)
                    .within_group(SqlSpan.duration_ns / 1000000.0)
                    .label("p95"),
                    func.array_agg(distinct(SqlSpan.trace_id))[:10].label("sample_traces"),
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
                samples = result.sample_traces[:10] if result.sample_traces else []
                slow_tools.append(
                    SlowTool(
                        tool_span_name=result.name,
                        count=result.count,
                        median_latency_ms=result.median or 0.0,
                        p95_latency_ms=result.p95 or 0.0,
                        sample_trace_ids=samples,
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
                    func.group_concat(distinct(SqlSpan.trace_id)).label("sample_traces"),
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
                # Parse samples for SQLite/MySQL
                samples = result.sample_traces.split(",")[:10] if result.sample_traces else []
                # Use avg as approximation for median, max as approximation for p95
                slow_tools.append(
                    SlowTool(
                        tool_span_name=result.name,
                        count=result.count,
                        median_latency_ms=result.avg_latency or 0.0,
                        p95_latency_ms=result.max_latency or 0.0,
                        sample_trace_ids=samples,
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
            func.count(case((func.length(SqlTraceMetadata.value) < 50, 1), else_=None)).label(
                "minimal_count"
            ),
            func.count(SqlTraceMetadata.request_id).label("total"),
        ).filter(
            SqlTraceMetadata.request_id.in_(select(sampled_traces)),
            SqlTraceMetadata.key == "mlflow.traceOutputs",
        )

        minimal_result = minimal_query.first()
        minimal_pct = (
            (minimal_result.minimal_count / minimal_result.total * 100)
            if minimal_result.total > 0
            else 0.0
        )

        # Get sample trace IDs for minimal responses
        minimal_samples = (
            session.query(SqlTraceMetadata.request_id)
            .filter(
                SqlTraceMetadata.request_id.in_(select(sampled_traces)),
                SqlTraceMetadata.key == "mlflow.traceOutputs",
                func.length(SqlTraceMetadata.value) < 50,
            )
            .limit(10)
            .all()
        )
        minimal_sample_ids = [s[0] for s in minimal_samples]

        # Analyze quality issues (contains certain keywords)
        quality_indicators = ["sorry", "apologize", "not sure", "cannot confirm"]
        quality_conditions = []
        for indicator in quality_indicators:
            quality_conditions.append(func.lower(SqlTraceMetadata.value).like(f"%{indicator}%"))

        quality_query = session.query(
            func.count(case((or_(*quality_conditions), 1), else_=None)).label("issue_count"),
            func.count(SqlTraceMetadata.request_id).label("total"),
        ).filter(
            SqlTraceMetadata.request_id.in_(select(sampled_traces)),
            SqlTraceMetadata.key == "mlflow.traceOutputs",
        )

        quality_result = quality_query.first()
        quality_pct = (
            (quality_result.issue_count / quality_result.total * 100)
            if quality_result.total > 0
            else 0.0
        )

        # Get sample trace IDs for quality issues
        quality_samples = (
            session.query(SqlTraceMetadata.request_id)
            .filter(
                SqlTraceMetadata.request_id.in_(select(sampled_traces)),
                SqlTraceMetadata.key == "mlflow.traceOutputs",
                or_(*quality_conditions),
            )
            .limit(10)
            .all()
        )
        quality_sample_ids = [s[0] for s in quality_samples]

        # For rushed processing and verbosity, we need duration analysis
        # Using database-side percentile calculation
        duration_stats = self._get_duration_distribution_stats(session, sampled_traces)

        return {
            "minimal": QualityMetric(
                value=minimal_pct,
                description="Percentage of responses shorter than 50 characters",
                sample_trace_ids=minimal_sample_ids,
            ),
            "quality_issues": QualityMetric(
                value=quality_pct,
                description="Percentage of responses containing uncertainty or apology phrases",
                sample_trace_ids=quality_sample_ids,
            ),
            "rushed": QualityMetric(
                value=duration_stats.get("rushed_pct", 0.0),
                description="Percentage of traces processed faster than P10 execution time",
                sample_trace_ids=duration_stats.get("rushed_samples", []),
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
        if self.dialect == "postgresql":
            # Calculate P10 threshold and count of traces below it
            stats_query = session.query(
                func.percentile_cont(0.1)
                .within_group(SqlSpan.duration_ns / 1000000.0)
                .label("p10"),
                func.count(SqlSpan.trace_id).label("total"),
            ).filter(
                SqlSpan.trace_id.in_(select(sampled_traces)),
                SqlSpan.parent_span_id.is_(None),
                SqlSpan.duration_ns.isnot(None),
            )

            result = stats_query.first()

            if result.total == 0:
                return {"rushed_pct": 0.0, "rushed_samples": []}

            # Count traces below P10
            rushed_count = (
                session.query(func.count(SqlSpan.trace_id))
                .filter(
                    SqlSpan.trace_id.in_(select(sampled_traces)),
                    SqlSpan.parent_span_id.is_(None),
                    SqlSpan.duration_ns < result.p10 * 1000000,
                )
                .scalar()
                or 0
            )

            # Get sample rushed trace IDs
            rushed_samples = (
                session.query(SqlSpan.trace_id)
                .filter(
                    SqlSpan.trace_id.in_(select(sampled_traces)),
                    SqlSpan.parent_span_id.is_(None),
                    SqlSpan.duration_ns < result.p10 * 1000000,
                )
                .limit(10)
                .all()
            )
            rushed_sample_ids = [s[0] for s in rushed_samples]

            return {
                "rushed_pct": (rushed_count / result.total * 100),
                "rushed_samples": rushed_sample_ids,
            }
        else:
            # Simplified for non-PostgreSQL
            return {"rushed_pct": 0.0, "rushed_samples": []}

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
