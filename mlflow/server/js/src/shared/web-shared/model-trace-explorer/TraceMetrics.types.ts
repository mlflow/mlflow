/**
 * Types for the QueryTraceMetrics API
 * Based on mlflow/protos/service.proto and mlflow/tracing/constant.py
 */

/**
 * Keys for metrics on traces view type.
 */
export enum TraceMetricKey {
  TRACE_COUNT = 'trace_count',
  LATENCY = 'latency',
  INPUT_TOKENS = 'input_tokens',
  OUTPUT_TOKENS = 'output_tokens',
  TOTAL_TOKENS = 'total_tokens',
}

/**
 * Common percentile values for latency metrics
 */
export const P50 = 50;
export const P90 = 90;
export const P99 = 99;

/**
 * Get the key for accessing percentile values in the API response
 * @param percentile - The percentile value (e.g., 50, 90, 99)
 * @returns The key string in format "Pxx.0" (e.g., "P50.0", "P90.0", "P99.0")
 */
export const getPercentileKey = (percentile: number): string => `P${percentile.toFixed(1)}`;

/**
 * Time bucket dimension key, included in results when time_interval_seconds is specified.
 * Applies to all view types (traces, spans, assessments).
 */
export const TIME_BUCKET_DIMENSION_KEY = 'time_bucket';

/**
 * View type prefix for trace-level filter expressions.
 * Based on mlflow/tracing/constant.py TraceMetricSearchKey.VIEW_TYPE
 */
export const TRACE_FILTER_VIEW_TYPE = 'trace';

/**
 * Search key fields for trace metrics filter expressions.
 * Based on mlflow/tracing/constant.py TraceMetricSearchKey
 */
export enum TraceFilterKey {
  /** Status field key */
  STATUS = 'status',
  /** Tag field key */
  TAG = 'tag',
  /** Metadata field key */
  METADATA = 'metadata',
}

/**
 * Trace status values for filter expressions.
 */
export enum TraceStatus {
  OK = 'OK',
  ERROR = 'ERROR',
}

/**
 * Creates a trace filter expression string.
 * @param field - The field to filter on (e.g., TraceFilterKey.STATUS)
 * @param value - The value to match (e.g., TraceStatus.ERROR)
 * @returns Filter expression string (e.g., 'trace.status = "ERROR"')
 */
export const createTraceFilter = (field: TraceFilterKey, value: string): string =>
  `${TRACE_FILTER_VIEW_TYPE}.${field} = "${value}"`;

/**
 * Keys for metrics on assessments view type.
 * Based on mlflow/tracing/constant.py AssessmentMetricKey
 */
export enum AssessmentMetricKey {
  /** Count of assessments */
  ASSESSMENT_COUNT = 'assessment_count',
  /** Numeric assessment value */
  ASSESSMENT_VALUE = 'assessment_value',
}

/**
 * View type prefix for assessment-level filter expressions.
 * Based on mlflow/tracing/constant.py AssessmentMetricSearchKey.VIEW_TYPE
 */
export const ASSESSMENT_FILTER_VIEW_TYPE = 'assessment';

/**
 * Search key fields for assessment metrics filter expressions.
 * Based on mlflow/tracing/constant.py AssessmentMetricSearchKey
 */
export enum AssessmentFilterKey {
  /** Assessment name field */
  NAME = 'name',
  /** Assessment type field (feedback, expectation) */
  TYPE = 'type',
}

/**
 * Assessment type values for filter expressions.
 */
export enum AssessmentTypeValue {
  FEEDBACK = 'feedback',
  EXPECTATION = 'expectation',
}

/**
 * Creates an assessment filter expression string.
 * @param field - The field to filter on (e.g., AssessmentFilterKey.NAME)
 * @param value - The value to match (e.g., "Correctness")
 * @returns Filter expression string (e.g., 'assessment.name = "Correctness"')
 */
export const createAssessmentFilter = (field: AssessmentFilterKey, value: string): string =>
  `${ASSESSMENT_FILTER_VIEW_TYPE}.${field} = "${value}"`;

/**
 * Dimension keys for assessment metrics.
 * Based on mlflow/tracing/constant.py AssessmentMetricDimensionKey
 */
export enum AssessmentDimensionKey {
  /** Assessment name dimension */
  ASSESSMENT_NAME = 'assessment_name',
  /** Assessment value dimension */
  ASSESSMENT_VALUE = 'assessment_value',
}

/**
 * Keys for metrics on spans view type.
 * Based on mlflow/tracing/constant.py SpanMetricKey
 */
export const SpanMetricKey = {
  /** Count of spans */
  SPAN_COUNT: 'span_count',
  /** Span latency in milliseconds */
  LATENCY: 'latency',
} as const;

export type SpanMetricKeyType = typeof SpanMetricKey[keyof typeof SpanMetricKey];

/**
 * View type prefix for span-level filter expressions.
 * Based on mlflow/tracing/constant.py SpanMetricSearchKey.VIEW_TYPE
 */
export const SPAN_FILTER_VIEW_TYPE = 'span';

/**
 * Search key fields for span metrics filter expressions.
 * Based on mlflow/tracing/constant.py SpanMetricSearchKey
 */
export const SpanFilterKey = {
  /** Span name field */
  NAME: 'name',
  /** Span status field (OK, ERROR) */
  STATUS: 'status',
  /** Span type field (LLM, TOOL, AGENT, etc.) */
  TYPE: 'type',
} as const;

/**
 * Span type values for filter expressions.
 * Based on mlflow/entities/span.py SpanType
 */
export const SpanType = {
  LLM: 'LLM',
  CHAIN: 'CHAIN',
  AGENT: 'AGENT',
  TOOL: 'TOOL',
  CHAT_MODEL: 'CHAT_MODEL',
  RETRIEVER: 'RETRIEVER',
  PARSER: 'PARSER',
  EMBEDDING: 'EMBEDDING',
  RERANKER: 'RERANKER',
  MEMORY: 'MEMORY',
  UNKNOWN: 'UNKNOWN',
  WORKFLOW: 'WORKFLOW',
  TASK: 'TASK',
  GUARDRAIL: 'GUARDRAIL',
  EVALUATOR: 'EVALUATOR',
} as const;

/**
 * Span status values for filter expressions.
 */
export const SpanStatus = {
  OK: 'OK',
  ERROR: 'ERROR',
  UNSET: 'UNSET',
} as const;

/**
 * Creates a span filter expression string.
 * @param field - The field to filter on (e.g., SpanFilterKey.TYPE)
 * @param value - The value to match (e.g., SpanType.TOOL)
 * @returns Filter expression string (e.g., 'span.type = "TOOL"')
 */
export const createSpanFilter = (field: string, value: string): string =>
  `${SPAN_FILTER_VIEW_TYPE}.${field} = "${value}"`;

/**
 * Dimension keys for span metrics.
 * Based on mlflow/tracing/constant.py SpanMetricDimensionKey
 */
export const SpanDimensionKey = {
  /** Span name dimension */
  SPAN_NAME: 'span_name',
  /** Span type dimension */
  SPAN_TYPE: 'span_type',
  /** Span status dimension */
  SPAN_STATUS: 'span_status',
} as const;

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
