"""Databricks SQL implementation of InsightsStore for trace and span analytics."""

import logging
from datetime import datetime

from mlflow.exceptions import MlflowException
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
from mlflow.insights.store.census_queries import (
    get_combined_basics_query,
    get_quality_metrics_query,
    get_spans_analysis_query,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.tracking.databricks_sql_store import DatabricksSqlStore

_logger = logging.getLogger(__name__)


class DatabricksSqlInsightsStore(DatabricksSqlStore, InsightsStore):
    """
    Databricks SQL implementation of InsightsStore for analyzing trace and span data.

    This class inherits from DatabricksSqlStore to get SQL capabilities and REST API operations,
    and implements InsightsStore interface for census generation.
    Uses Databricks SQL for efficient trace data queries.
    """

    def __init__(self, store_uri="databricks"):
        """
        Initialize DatabricksSqlInsightsStore with Databricks connection.

        Args:
            store_uri: Databricks URI (e.g., 'databricks' or 'databricks://<profile>')
        """
        # Initialize parent DatabricksSqlStore
        super().__init__(store_uri)

    def generate_census(self, experiment_id: str) -> Census:
        """
        Generate a comprehensive census of trace data from Databricks.

        Analyzes traces to provide statistical distributions and patterns
        including operational metrics and quality assessments.

        Args:
            experiment_id: The experiment ID to analyze

        Returns:
            Census object containing statistics and distributions
        """

        # Validate experiment_id
        if not experiment_id:
            raise MlflowException(
                "experiment_id parameter is required.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Get trace table name for the experiment
        experiment_id = experiment_id.strip()
        _logger.debug(f"Getting trace table for experiment {experiment_id}")
        table_name = self._get_trace_table_for_experiment(experiment_id)

        if not table_name:
            raise MlflowException(
                f"No trace table found for experiment {experiment_id}. "
                f"This experiment may not have tracing enabled.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        _logger.debug(f"Using trace table: {table_name}")

        # Get queries from the census_queries module
        combined_basics_query = get_combined_basics_query(table_name)
        spans_query = get_spans_analysis_query(table_name)
        quality_query = get_quality_metrics_query(table_name)

        # Execute 3 optimized queries
        basics_results = self.execute_sql(combined_basics_query)
        spans_results = self.execute_sql(spans_query)
        quality_results = self.execute_sql(quality_query)

        # Process results
        basics = basics_results[0]
        quality = quality_results[0] if quality_results else {}

        # Parse timestamp helper
        def parse_timestamp(ts_val):
            if isinstance(ts_val, str):
                # Try ISO format first, then fall back to simpler format
                try:
                    return datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                except Exception:
                    try:
                        return datetime.strptime(ts_val, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return datetime.now()
            elif isinstance(ts_val, datetime):
                return ts_val
            else:
                return datetime.now()

        # Clean sample IDs helper - removes None values
        def clean_sample_ids(sample_ids):
            if not sample_ids:
                return []
            return [id for id in sample_ids if id is not None]

        # Process spans results
        error_span_entities = []
        slow_tool_entities = []
        for row in spans_results:
            if row.get("type") == "error" and len(error_span_entities) < 5:
                error_span_entities.append(
                    ErrorSpan(
                        error_span_name=row.get("name", ""),
                        count=row.get("count", 0),
                        pct_of_errors=row.get("percentage", 0.0),
                        sample_trace_ids=clean_sample_ids(row.get("sample_trace_ids", [])),
                    )
                )
            elif row.get("type") == "slow_tool" and len(slow_tool_entities) < 5:
                slow_tool_entities.append(
                    SlowTool(
                        tool_span_name=row.get("name", ""),
                        count=row.get("count", 0),
                        median_latency_ms=row.get("median_latency_ms", 0.0),
                        p95_latency_ms=row.get("p95_latency_ms", 0.0),
                        sample_trace_ids=clean_sample_ids(row.get("sample_trace_ids", [])),
                    )
                )

        # Build operational metrics
        operational_metrics = OperationalMetrics(
            total_traces=basics.get("total_traces", 0),
            ok_count=basics.get("ok_count", 0),
            error_count=basics.get("error_count", 0),
            error_rate=basics.get("error_rate_percentage", 0.0),
            # SQL returns NULL for these when no OK traces exist which will get converted to None
            p50_latency_ms=basics.get("p50_latency_ms"),
            p90_latency_ms=basics.get("p90_latency_ms"),
            p95_latency_ms=basics.get("p95_latency_ms"),
            p99_latency_ms=basics.get("p99_latency_ms"),
            max_latency_ms=basics.get("max_latency_ms"),
            first_trace_timestamp=parse_timestamp(basics.get("first_trace_timestamp")),
            last_trace_timestamp=parse_timestamp(basics.get("last_trace_timestamp")),
            time_buckets=[],
            top_error_spans=error_span_entities,
            top_slow_tools=slow_tool_entities,
        )

        # Build quality metrics
        # Use None if value is not available
        quality_metrics = QualityMetrics(
            minimal_responses=QualityMetric(
                value=quality.get("minimal_response_rate"),
                description="Percentage of responses shorter than 50 characters",
                sample_trace_ids=clean_sample_ids(quality.get("minimal_sample_ids", [])),
            ),
            response_quality_issues=QualityMetric(
                value=quality.get("problematic_response_rate"),
                description=(
                    "Percentage of responses containing uncertainty or apology phrases "
                    "(?, 'apologize', 'sorry', 'not sure', 'cannot confirm')"
                ),
                sample_trace_ids=clean_sample_ids(quality.get("quality_sample_ids", [])),
            ),
            rushed_processing=QualityMetric(
                value=quality.get("rushed_complex_pct"),
                description=(
                    "Percentage of complex requests (>P75 request length) "
                    "processed quickly (<P10 execution time)"
                ),
                sample_trace_ids=clean_sample_ids(quality.get("rushed_sample_ids", [])),
            ),
            verbosity=QualityMetric(
                value=quality.get("verbose_percentage"),
                description=(
                    "Percentage of short inputs (≤P25 request length) "
                    "receiving verbose responses (>P90 response length)"
                ),
                sample_trace_ids=clean_sample_ids(quality.get("verbosity_sample_ids", [])),
            ),
        )

        return Census(
            metadata=CensusMetadata(
                created_at=datetime.now(),
                table_name=table_name,
                additional_metadata={
                    "experiment_id": experiment_id,
                    "backend": "databricks",
                },
            ),
            operational_metrics=operational_metrics,
            quality_metrics=quality_metrics,
        )
