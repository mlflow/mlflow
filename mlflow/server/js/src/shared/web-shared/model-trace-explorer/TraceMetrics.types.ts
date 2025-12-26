/**
 * Types for the QueryTraceMetrics API
 * Based on mlflow/protos/service.proto and mlflow/tracing/constant.py
 */

/**
 * Keys for metrics on traces view type.
 */
export const TraceMetricKey = {
  TRACE_COUNT: 'trace_count',
  LATENCY: 'latency',
  INPUT_TOKENS: 'input_tokens',
  OUTPUT_TOKENS: 'output_tokens',
  TOTAL_TOKENS: 'total_tokens',
} as const;

export type TraceMetricKeyType = typeof TraceMetricKey[keyof typeof TraceMetricKey];

/**
 * Time bucket dimension key, included in results when time_interval_seconds is specified.
 * Applies to all view types (traces, spans, assessments).
 */
export const TIME_BUCKET_DIMENSION_KEY = 'time_bucket';

/**
 * The level at which to aggregate metrics.
 */
export enum MetricViewType {
  /** Aggregate at trace level */
  TRACES = 'TRACES',
  /** Aggregate at span level */
  SPANS = 'SPANS',
  /** Aggregate at assessment level */
  ASSESSMENTS = 'ASSESSMENTS',
}

/**
 * Aggregation type for metrics.
 */
export enum AggregationType {
  /** Count of entities */
  COUNT = 'COUNT',
  /** Sum of values */
  SUM = 'SUM',
  /** Average of values */
  AVG = 'AVG',
  /** Percentile aggregation (requires percentile_value parameter) */
  PERCENTILE = 'PERCENTILE',
  /** Minimum value */
  MIN = 'MIN',
  /** Maximum value */
  MAX = 'MAX',
}

/**
 * Aggregation specification for a metric query.
 */
export interface MetricAggregation {
  /** The type of aggregation to apply */
  aggregation_type: AggregationType;
  /**
   * The percentile value to compute (0-100), required when aggregation_type is PERCENTILE.
   * Examples: 50 (median), 75, 90, 95, 99.
   * This field is ignored for other aggregation types.
   */
  percentile_value?: number;
}

/**
 * Request payload for QueryTraceMetrics API.
 */
export interface QueryTraceMetricsRequest {
  /** Required: The experiment IDs to search traces */
  experiment_ids: string[];
  /** Required: The level at which to aggregate metrics */
  view_type: MetricViewType;
  /** Required: The name of the metric to query (e.g. "latency") */
  metric_name: string;
  /** Required: The aggregations to apply */
  aggregations: MetricAggregation[];
  /** Optional: Dimensions to group metrics by (e.g. "name", "status") */
  dimensions?: string[];
  /** Optional: Filter expressions to apply (e.g. `trace.status="OK"`) */
  filters?: string[];
  /**
   * Optional: Time interval for grouping in seconds.
   * When set, results automatically include a time dimension grouped by the specified interval.
   * Examples: 60 (minute), 3600 (hour), 86400 (day), 604800 (week), 2592000 (month).
   */
  time_interval_seconds?: number;
  /** Optional: Start of time range in milliseconds since epoch. Required if time_interval_seconds is set. */
  start_time_ms?: number;
  /** Optional: End of time range in milliseconds since epoch. Required if time_interval_seconds is set. */
  end_time_ms?: number;
  /** Optional: Maximum number of data points to return. Default: 1000 */
  max_results?: number;
  /** Optional: Pagination token for fetching the next page of results */
  page_token?: string;
}

/**
 * A single data point with dimension values and metric values.
 */
export interface MetricDataPoint {
  /** Metric name, e.g. "latency" */
  metric_name: string;
  /**
   * Dimension values for this data point.
   * Keys correspond to dimensions, e.g., {"status": "OK"}
   * When time_interval_seconds is specified, includes "time_bucket" with ISO timestamp string.
   */
  dimensions: Record<string, string>;
  /**
   * Metric values for this data point.
   * Keys are aggregation types, e.g., {"AVG": 150, "P99": 234.5}
   */
  values: Record<string, number>;
}

/**
 * Response from QueryTraceMetrics API.
 */
export interface QueryTraceMetricsResponse {
  /** Data points grouped by dimensions */
  data_points: MetricDataPoint[];
  /** Pagination token for fetching the next page. Empty if no more results are available. */
  next_page_token?: string;
}
