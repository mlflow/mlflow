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

        # 1. Basic counts
        basic_query = f"""
        SELECT
            COUNT(*) as total_traces,
            SUM(CASE WHEN state = 'OK' THEN 1 ELSE 0 END) as ok_count,
            SUM(CASE WHEN state = 'ERROR' THEN 1 ELSE 0 END) as error_count,
            ROUND(SUM(CASE WHEN state = 'ERROR' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                as error_rate_percentage
        FROM {table_name}
        """

        # 2. Latency percentiles
        latency_query = f"""
        SELECT
            percentile(execution_duration_ms, 0.5) as p50_latency_ms,
            percentile(execution_duration_ms, 0.9) as p90_latency_ms,
            percentile(execution_duration_ms, 0.95) as p95_latency_ms,
            percentile(execution_duration_ms, 0.99) as p99_latency_ms,
            MAX(execution_duration_ms) as max_latency_ms
        FROM {table_name}
        WHERE state = 'OK' AND execution_duration_ms IS NOT NULL
        """

        # 3. Top error spans with sample trace IDs
        error_query = f"""
        WITH error_spans_with_traces AS (
            SELECT
                span.name as error_span_name,
                t.trace_id,
                COUNT(*) OVER (PARTITION BY span.name) as count,
                ROUND(COUNT(*) OVER (PARTITION BY span.name) * 100.0 / (
                    SELECT COUNT(*)
                    FROM {table_name}
                    LATERAL VIEW explode(spans) AS s
                    WHERE s.status_code = 'ERROR'
                ), 2) as percentage_of_errors,
                ROW_NUMBER() OVER (PARTITION BY span.name ORDER BY t.trace_id) as rn
            FROM {table_name} t
            LATERAL VIEW explode(spans) AS span
            WHERE span.status_code = 'ERROR'
        ),
        error_spans_summary AS (
            SELECT
                error_span_name,
                count,
                percentage_of_errors,
                collect_list(CASE WHEN rn <= 15 THEN trace_id END) as sample_trace_ids
            FROM error_spans_with_traces
            GROUP BY error_span_name, count, percentage_of_errors
        )
        SELECT
            error_span_name,
            count,
            percentage_of_errors,
            sample_trace_ids
        FROM error_spans_summary
        ORDER BY count DESC
        LIMIT 5
        """

        # 4. Top slow tools with sample trace IDs
        slow_tools_query = f"""
        WITH slow_tools_with_traces AS (
            SELECT
                span.name as tool_span_name,
                t.trace_id,
                (unix_timestamp(span.end_time) - unix_timestamp(span.start_time)) * 1000
                    as latency_ms,
                COUNT(*) OVER (PARTITION BY span.name) as count,
                percentile((unix_timestamp(span.end_time) - unix_timestamp(span.start_time)) * 1000,
                    0.95) OVER (PARTITION BY span.name) as p95_latency_ms,
                percentile((unix_timestamp(span.end_time) - unix_timestamp(span.start_time)) * 1000,
                    0.5) OVER (PARTITION BY span.name) as median_latency_ms,
                ROW_NUMBER() OVER (PARTITION BY span.name
                    ORDER BY (unix_timestamp(span.end_time) - unix_timestamp(span.start_time))
                        * 1000 DESC) as rn
            FROM {table_name} t
            LATERAL VIEW explode(spans) AS span
            WHERE span.start_time IS NOT NULL AND span.end_time IS NOT NULL
        ),
        slow_tools_summary AS (
            SELECT
                tool_span_name,
                count,
                p95_latency_ms,
                median_latency_ms,
                collect_list(CASE WHEN rn <= 15 THEN trace_id END) as sample_trace_ids
            FROM slow_tools_with_traces
            GROUP BY tool_span_name, count, p95_latency_ms, median_latency_ms
            HAVING count >= 10
        )
        SELECT
            tool_span_name,
            count,
            median_latency_ms,
            p95_latency_ms,
            sample_trace_ids
        FROM slow_tools_summary
        ORDER BY p95_latency_ms DESC
        LIMIT 5
        """

        # 5. Time buckets - removed for now, returning empty array
        # TODO: Implement smart time buckets that highlight anomalies

        # 6. Timestamp range
        timestamp_query = f"""
        SELECT
            MIN(request_time) as first_trace_timestamp,
            MAX(request_time) as last_trace_timestamp
        FROM {table_name}
        """

        # 7. Quality Metrics - Verbosity Analysis
        verbosity_query = f"""
        WITH percentile_thresholds AS (
          SELECT
            percentile(LENGTH(request), 0.25) as short_input_threshold,
            percentile(LENGTH(response), 0.90) as verbose_response_threshold
          FROM {table_name}
          WHERE state = 'OK'
        ),
        shorter_inputs AS (
          SELECT
            t.trace_id,
            LENGTH(t.response) as response_length
          FROM {table_name} t
          CROSS JOIN percentile_thresholds p
          WHERE t.state = 'OK'
            AND LENGTH(t.request) <= p.short_input_threshold
        ),
        verbose_traces AS (
          SELECT
            trace_id,
            response_length > (SELECT verbose_response_threshold FROM percentile_thresholds)
                as is_verbose
          FROM shorter_inputs
        ),
        limited_samples AS (
          SELECT
            trace_id,
            is_verbose,
            ROW_NUMBER() OVER (PARTITION BY is_verbose ORDER BY trace_id) as rn
          FROM verbose_traces
        )
        SELECT
          ROUND(100.0 * SUM(CASE WHEN is_verbose THEN 1 ELSE 0 END) / COUNT(*), 2)
              as verbose_percentage,
          collect_list(CASE WHEN is_verbose AND rn <= 15 THEN trace_id END) as sample_trace_ids
        FROM limited_samples
        """

        # 8. Quality Metrics - Rushed Processing
        rushed_processing_query = f"""
        WITH percentile_thresholds AS (
          SELECT
            percentile(LENGTH(request), 0.75) as complex_threshold,
            percentile(execution_duration_ms, 0.10) as fast_threshold
          FROM {table_name}
          WHERE state = 'OK' AND execution_duration_ms > 0
        ),
        complex_requests AS (
          SELECT
            t.trace_id,
            LENGTH(t.request) > p.complex_threshold as is_complex,
            t.execution_duration_ms < p.fast_threshold as is_fast
          FROM {table_name} t
          CROSS JOIN percentile_thresholds p
          WHERE t.state = 'OK' AND t.execution_duration_ms > 0
        ),
        limited_samples AS (
          SELECT
            trace_id,
            is_complex,
            is_fast,
            ROW_NUMBER() OVER (PARTITION BY (is_complex AND is_fast) ORDER BY trace_id) as rn
          FROM complex_requests
        )
        SELECT
          ROUND(100.0 * SUM(CASE WHEN is_complex AND is_fast THEN 1 ELSE 0 END) /
              NULLIF(SUM(CASE WHEN is_complex THEN 1 ELSE 0 END), 0), 2)
              as rushed_complex_pct,
          collect_list(CASE WHEN is_complex AND is_fast AND rn <= 15 THEN trace_id END)
              as sample_trace_ids
        FROM limited_samples
        """

        # 9. Quality Metrics - Minimal Responses
        minimal_responses_query = f"""
        WITH minimal_check AS (
          SELECT
            trace_id,
            LENGTH(response) < 50 as is_minimal
          FROM {table_name}
          WHERE state = 'OK'
        ),
        limited_samples AS (
          SELECT
            trace_id,
            is_minimal,
            ROW_NUMBER() OVER (PARTITION BY is_minimal ORDER BY trace_id) as rn
          FROM minimal_check
        )
        SELECT
          ROUND(100.0 * SUM(CASE WHEN is_minimal THEN 1 ELSE 0 END) / COUNT(*), 2)
              as minimal_response_rate,
          collect_list(CASE WHEN is_minimal AND rn <= 15 THEN trace_id END) as sample_trace_ids
        FROM limited_samples
        """

        # 10. Quality Metrics - Response Quality Issues
        response_quality_query = f"""
        WITH quality_issues AS (
          SELECT
            trace_id,
            (response LIKE '%?%' OR
             LOWER(response) LIKE '%apologize%' OR LOWER(response) LIKE '%sorry%' OR
             LOWER(response) LIKE '%not sure%' OR LOWER(response) LIKE '%cannot confirm%')
                as has_quality_issue
          FROM {table_name}
          WHERE state = 'OK'
        ),
        limited_samples AS (
          SELECT
            trace_id,
            has_quality_issue,
            ROW_NUMBER() OVER (PARTITION BY has_quality_issue ORDER BY trace_id) as rn
          FROM quality_issues
        )
        SELECT
          ROUND(100.0 * SUM(CASE WHEN has_quality_issue THEN 1 ELSE 0 END) / COUNT(*), 2)
              as problematic_response_rate_percentage,
          collect_list(CASE WHEN has_quality_issue AND rn <= 15 THEN trace_id END)
              as sample_trace_ids
        FROM limited_samples
        """

        # Execute queries
        basic_results = self.execute_sql(basic_query)
        latency_results = self.execute_sql(latency_query)
        error_results = self.execute_sql(error_query)
        slow_tools_results = self.execute_sql(slow_tools_query)
        timestamp_results = self.execute_sql(timestamp_query)
        verbosity_results = self.execute_sql(verbosity_query)
        rushed_processing_results = self.execute_sql(rushed_processing_query)
        minimal_responses_results = self.execute_sql(minimal_responses_query)
        response_quality_results = self.execute_sql(response_quality_query)

        # Extract results
        basic = basic_results[0] if basic_results else {}
        latency = latency_results[0] if latency_results else {}
        timestamps = timestamp_results[0] if timestamp_results else {}
        verbosity = verbosity_results[0] if verbosity_results else {}
        rushed_processing = rushed_processing_results[0] if rushed_processing_results else {}
        minimal_responses = minimal_responses_results[0] if minimal_responses_results else {}
        response_quality = response_quality_results[0] if response_quality_results else {}

        # Build time bucket entities - empty for now
        time_bucket_entities = []  # TODO: Implement smart time buckets

        # Build error span entities
        error_span_entities = []
        for error in error_results[:5]:  # Limit to 5
            # Clean up sample_trace_ids - filter out None values
            sample_ids = error.get("sample_trace_ids", [])
            if sample_ids:
                sample_ids = [id for id in sample_ids if id is not None]

            error_span_entities.append(
                ErrorSpan(
                    error_span_name=error.get("error_span_name", ""),
                    count=error.get("count", 0),
                    pct_of_errors=error.get("percentage_of_errors", 0.0),
                    sample_trace_ids=sample_ids,
                )
            )

        # Build slow tool entities
        slow_tool_entities = []
        for tool in slow_tools_results[:5]:  # Limit to 5
            # Clean up sample_trace_ids - filter out None values
            sample_ids = tool.get("sample_trace_ids", [])
            if sample_ids:
                sample_ids = [id for id in sample_ids if id is not None]

            slow_tool_entities.append(
                SlowTool(
                    tool_span_name=tool.get("tool_span_name", ""),
                    count=tool.get("count", 0),
                    median_latency_ms=tool.get("median_latency_ms", 0.0),
                    p95_latency_ms=tool.get("p95_latency_ms", 0.0),
                    sample_trace_ids=sample_ids,
                )
            )

        # Parse first and last trace timestamps
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

        # Build operational metrics entity
        operational_metrics = OperationalMetrics(
            total_traces=basic.get("total_traces", 0),
            ok_count=basic.get("ok_count", 0),
            error_count=basic.get("error_count", 0),
            error_rate=basic.get("error_rate_percentage", 0.0),
            p50_latency_ms=latency.get("p50_latency_ms", 0.0),
            p90_latency_ms=latency.get("p90_latency_ms", 0.0),
            p95_latency_ms=latency.get("p95_latency_ms", 0.0),
            p99_latency_ms=latency.get("p99_latency_ms", 0.0),
            max_latency_ms=latency.get("max_latency_ms", 0.0),
            first_trace_timestamp=parse_timestamp(timestamps.get("first_trace_timestamp")),
            last_trace_timestamp=parse_timestamp(timestamps.get("last_trace_timestamp")),
            time_buckets=time_bucket_entities,
            top_error_spans=error_span_entities,
            top_slow_tools=slow_tool_entities,
        )

        # Clean up sample_trace_ids for quality metrics
        def clean_sample_ids(sample_ids):
            if not sample_ids:
                return []
            return [id for id in sample_ids[:10] if id is not None]

        # Build quality metrics entity
        quality_metrics = QualityMetrics(
            minimal_responses=QualityMetric(
                value=minimal_responses.get("minimal_response_rate", 0.0),
                description="Percentage of responses shorter than 50 characters",
                sample_trace_ids=clean_sample_ids(minimal_responses.get("sample_trace_ids", [])),
            ),
            response_quality_issues=QualityMetric(
                value=response_quality.get("problematic_response_rate_percentage", 0.0),
                description="Percentage of responses containing uncertainty or apology phrases",
                sample_trace_ids=clean_sample_ids(response_quality.get("sample_trace_ids", [])),
            ),
            rushed_processing=QualityMetric(
                value=rushed_processing.get("rushed_complex_pct", 0.0),
                description="Percentage of traces processed faster than P10 execution time",
                sample_trace_ids=clean_sample_ids(rushed_processing.get("sample_trace_ids", [])),
            ),
            verbosity=QualityMetric(
                value=verbosity.get("verbose_percentage", 0.0),
                description="Percentage of short inputs receiving verbose responses",
                sample_trace_ids=clean_sample_ids(verbosity.get("sample_trace_ids", [])),
            ),
        )

        # Build census entity
        census = Census(
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

        # Return as JSON string using the model's serialization
        return census
