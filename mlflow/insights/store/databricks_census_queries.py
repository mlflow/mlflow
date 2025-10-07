"""
SQL queries for Census generation in Databricks SQL Insights Store.

This module contains all the SQL query templates used by the DatabricksSqlInsightsStore
for generating census reports. The queries are separated by their purpose and optimized
for performance on Databricks SQL.
"""


def get_combined_basics_query(table_name: str) -> str:
    """
    Query for basic metrics, latency percentiles, and timestamp ranges.
    Combines multiple simple aggregations into a single query for efficiency.
    Always returns exactly one row with default values when table is empty.

    Args:
        table_name: The fully qualified table name to query

    Returns:
        SQL query string for basic operational metrics
    """
    return f"""
    SELECT
        COUNT(*) as total_traces,
        SUM(CASE WHEN state = 'OK' THEN 1 ELSE 0 END) as ok_count,
        SUM(CASE WHEN state = 'ERROR' THEN 1 ELSE 0 END) as error_count,
        ROUND(
            SUM(CASE WHEN state = 'ERROR' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2
        ) as error_rate_percentage,
        MIN(request_time) as first_trace_timestamp,
        MAX(request_time) as last_trace_timestamp,
        percentile(CASE WHEN state = 'OK' THEN execution_duration_ms END, 0.5) as p50_latency_ms,
        percentile(CASE WHEN state = 'OK' THEN execution_duration_ms END, 0.9) as p90_latency_ms,
        percentile(CASE WHEN state = 'OK' THEN execution_duration_ms END, 0.95) as p95_latency_ms,
        percentile(CASE WHEN state = 'OK' THEN execution_duration_ms END, 0.99) as p99_latency_ms,
        MAX(CASE WHEN state = 'OK' THEN execution_duration_ms END) as max_latency_ms
    FROM {table_name}
    """


def get_spans_analysis_query(table_name: str) -> str:
    """
    Query for analyzing span-level data including errors and slow tools.
    Combines error span analysis and slow tool detection using a single scan
    of exploded spans for efficiency.

    Args:
        table_name: The fully qualified table name to query

    Returns:
        SQL query string for span analysis (errors and slow tools)
    """
    return f"""
    WITH exploded_spans AS (
        SELECT
            t.trace_id,
            span.name as span_name,
            span.status_code as span_status_code,
            (unix_timestamp(span.end_time) - unix_timestamp(span.start_time)) * 1000
                as span_latency_ms
        FROM {table_name} t
        LATERAL VIEW explode(spans) AS span
    ),
    error_spans AS (
        SELECT
            span_name as error_span_name,
            COUNT(*) as count,
            collect_list(trace_id) as trace_ids
        FROM exploded_spans
        WHERE span_status_code = 'ERROR'
        GROUP BY span_name
        ORDER BY count DESC
        LIMIT 5
    ),
    slow_tools AS (
        SELECT
            span_name as tool_span_name,
            COUNT(*) as count,
            percentile(span_latency_ms, 0.5) as median_latency_ms,
            percentile(span_latency_ms, 0.95) as p95_latency_ms,
            collect_list(trace_id) as trace_ids
        FROM exploded_spans
        WHERE span_latency_ms IS NOT NULL
        GROUP BY span_name
        HAVING count >= 10
        ORDER BY p95_latency_ms DESC
        LIMIT 5
    ),
    total_errors AS (
        SELECT COUNT(*) as total_error_spans
        FROM exploded_spans
        WHERE span_status_code = 'ERROR'
    )
    SELECT
        'error' as type,
        e.error_span_name as name,
        e.count,
        ROUND(e.count * 100.0 / t.total_error_spans, 2) as percentage,
        NULL as median_latency_ms,
        NULL as p95_latency_ms,
        slice(e.trace_ids, 1, 10) as sample_trace_ids
    FROM error_spans e, total_errors t
    UNION ALL
    SELECT
        'slow_tool' as type,
        s.tool_span_name as name,
        s.count,
        NULL as percentage,
        s.median_latency_ms,
        s.p95_latency_ms,
        slice(s.trace_ids, 1, 10) as sample_trace_ids
    FROM slow_tools s
    """


def get_quality_metrics_query(table_name: str) -> str:
    """
    Query for all quality metrics including minimal responses, quality issues,
    verbosity, and rushed processing detection. Combines multiple quality checks
    into a single query scanning OK traces once.

    Args:
        table_name: The fully qualified table name to query

    Returns:
        SQL query string for quality metrics analysis
    """
    return f"""
    WITH ok_traces AS (
        SELECT
            trace_id,
            request,
            response,
            execution_duration_ms,
            LENGTH(request) as request_length,
            LENGTH(response) as response_length
        FROM {table_name}
        WHERE state = 'OK'
    ),
    thresholds AS (
        SELECT
            percentile(request_length, 0.25) as short_input_threshold,
            percentile(request_length, 0.75) as complex_threshold,
            percentile(response_length, 0.90) as verbose_response_threshold,
            percentile(execution_duration_ms, 0.10) as fast_threshold
        FROM ok_traces
        WHERE execution_duration_ms > 0
    ),
    trace_flags AS (
        SELECT
            t.trace_id,
            t.response_length < 50 as is_minimal,
            (t.response LIKE '%?%' OR LOWER(t.response) LIKE '%apologize%'
             OR LOWER(t.response) LIKE '%sorry%'
             OR LOWER(t.response) LIKE '%not sure%'
             OR LOWER(t.response) LIKE '%cannot confirm%') as has_quality_issue,
            t.request_length <= th.short_input_threshold
                AND t.response_length > th.verbose_response_threshold
                as is_verbose_short_input,
            t.request_length > th.complex_threshold
                AND t.execution_duration_ms < th.fast_threshold
                as is_rushed_complex
        FROM ok_traces t
        CROSS JOIN thresholds th
    )
    SELECT
        ROUND(
            100.0 * SUM(CASE WHEN is_minimal THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2
        ) as minimal_response_rate,
        ROUND(
            100.0 * SUM(CASE WHEN has_quality_issue THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        ) as problematic_response_rate,
        ROUND(
            100.0 * SUM(CASE WHEN is_verbose_short_input THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        ) as verbose_percentage,
        ROUND(
            100.0 * SUM(CASE WHEN is_rushed_complex THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        ) as rushed_complex_pct,
        slice(collect_list(CASE WHEN is_minimal THEN trace_id END), 1, 10)
            as minimal_sample_ids,
        slice(collect_list(CASE WHEN has_quality_issue THEN trace_id END), 1, 10)
            as quality_sample_ids,
        slice(
            collect_list(CASE WHEN is_verbose_short_input THEN trace_id END), 1, 10
        ) as verbosity_sample_ids,
        slice(collect_list(CASE WHEN is_rushed_complex THEN trace_id END), 1, 10)
            as rushed_sample_ids
    FROM trace_flags
    """
