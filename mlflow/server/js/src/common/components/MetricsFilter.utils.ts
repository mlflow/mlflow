import {
  FilterOperator,
  GIT_BRANCH_COLUMN_ID,
  GIT_COMMIT_COLUMN_ID,
  SESSION_COLUMN_ID,
  STATE_COLUMN_ID,
  USER_COLUMN_ID,
} from '@databricks/web-shared/genai-traces-table';
import {
  MLFLOW_GIT_BRANCH_KEY,
  MLFLOW_GIT_COMMIT_KEY,
  MLFLOW_TRACE_USER_KEY,
  SESSION_ID_METADATA_KEY,
  TraceFilterKey,
  TraceStatus,
  createTraceFilter,
  createTraceMetadataFilter,
} from '@databricks/web-shared/model-trace-explorer';

/**
 * Curated list of columns the metrics API can filter on with `=`.
 * Add a new entry here AND in `translateToMetricsFilters` and
 * `translateToTracesPageFilters` to expose more dimensions
 */
export type MetricFilterColumn = 'user' | 'session' | 'state' | 'git_branch' | 'git_commit';

export interface MetricFilter {
  column: MetricFilterColumn;
  value: string;
}

export interface MetricFilterColumnOption {
  value: MetricFilterColumn;
  label: string;
  valueOptions?: { value: string; label: string }[];
}

export const TRACE_STATE_VALUES = [TraceStatus.IN_PROGRESS, TraceStatus.OK, TraceStatus.ERROR] as const;

export const isCompleteFilter = (filter: MetricFilter): boolean => Boolean(filter.column) && Boolean(filter.value);

/**
 * Builder for each column's metrics-API DSL filter string. Each entry is a
 * closure so columns can pick *which* helper to call (most use
 * `createTraceMetadataFilter` with a metadata key; `state` uses
 * `createTraceFilter` with a `TraceFilterKey`) without forcing the mapping
 * into a uniform shape.
 *
 * The `Record<MetricFilterColumn, ...>` type makes adding a new
 * MetricFilterColumn a compile error until the builder is filled in.
 */
const COLUMN_TO_METRICS_FILTER_BUILDER: Record<MetricFilterColumn, (value: string) => string> = {
  user: (v) => createTraceMetadataFilter(MLFLOW_TRACE_USER_KEY, v),
  session: (v) => createTraceMetadataFilter(SESSION_ID_METADATA_KEY, v),
  state: (v) => createTraceFilter(TraceFilterKey.STATUS, v),
  git_branch: (v) => createTraceMetadataFilter(MLFLOW_GIT_BRANCH_KEY, v),
  git_commit: (v) => createTraceMetadataFilter(MLFLOW_GIT_COMMIT_KEY, v),
};

/**
 * Translates user-driven filter rows from MetricsFilter into metrics-API DSL
 * filter strings consumed by useTraceMetricsQuery via OverviewChartProvider.
 */
export const translateToMetricsFilters = (filters: MetricFilter[]): string[] | undefined => {
  const result = filters.filter(isCompleteFilter).map((f) => COLUMN_TO_METRICS_FILTER_BUILDER[f.column](f.value));
  return result.length > 0 ? result : undefined;
};

/**
 * Mapping from MetricFilter column to the corresponding traces-table column id,
 * used by `translateToTracesPageFilters` to forward filters via URL params.
 *
 * The `Record<MetricFilterColumn, ...>` type makes adding a new
 * MetricFilterColumn a compile error until the mapping is filled in.
 */
const COLUMN_TO_TRACES_COLUMN_ID: Record<MetricFilterColumn, string> = {
  user: USER_COLUMN_ID,
  session: SESSION_COLUMN_ID,
  state: STATE_COLUMN_ID,
  git_branch: GIT_BRANCH_COLUMN_ID,
  git_commit: GIT_COMMIT_COLUMN_ID,
};

/**
 * Translates user-driven filter rows from MetricsFilter into Traces page URL
 * filter strings consumed by useFilters on the Traces tab. Used to forward
 * overview filters when navigating from chart tooltip "View traces" links to
 * the Traces page.
 *
 * URL filter format: `column::operator::value[::key]`. The trailing `::key`
 * segment is optional and only used for filters that disambiguate within a
 * column group (e.g. assessment filters); top-level columns like `user` emit
 * the 3-segment form.
 *
 */
export const translateToTracesPageFilters = (filters: MetricFilter[]): string[] | undefined => {
  const result = filters
    .filter(isCompleteFilter)
    .map((f) => [COLUMN_TO_TRACES_COLUMN_ID[f.column], FilterOperator.EQUALS, f.value].join('::'));
  return result.length > 0 ? result : undefined;
};
