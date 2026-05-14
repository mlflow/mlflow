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
 * `translateToTracesPageFilters` to expose more dimensions.
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
 * Translates user-driven filter rows from MetricsFilter into metrics-API DSL
 * filter strings consumed by useTraceMetricsQuery via OverviewChartProvider.
 *
 * Add a new case here when adding a new column option in MetricsFilter.
 */
export const translateToMetricsFilters = (filters: MetricFilter[]): string[] | undefined => {
  const result = filters
    .map((f) => {
      if (!f.column || !f.value) return null;
      switch (f.column) {
        case 'user':
          return createTraceMetadataFilter(MLFLOW_TRACE_USER_KEY, f.value);
        case 'session':
          return createTraceMetadataFilter(SESSION_ID_METADATA_KEY, f.value);
        case 'state':
          return createTraceFilter(TraceFilterKey.STATUS, f.value);
        case 'git_branch':
          return createTraceMetadataFilter(MLFLOW_GIT_BRANCH_KEY, f.value);
        case 'git_commit':
          return createTraceMetadataFilter(MLFLOW_GIT_COMMIT_KEY, f.value);
        default:
          return null;
      }
    })
    .filter((s): s is string => s !== null);
  return result.length > 0 ? result : undefined;
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
 * Add a new case here when adding a new column option in MetricsFilter.
 */
export const translateToTracesPageFilters = (filters: MetricFilter[]): string[] | undefined => {
  const result = filters
    .map((f) => {
      if (!f.column || !f.value) return null;
      switch (f.column) {
        case 'user':
          return [USER_COLUMN_ID, FilterOperator.EQUALS, f.value].join('::');
        case 'session':
          return [SESSION_COLUMN_ID, FilterOperator.EQUALS, f.value].join('::');
        case 'state':
          return [STATE_COLUMN_ID, FilterOperator.EQUALS, f.value].join('::');
        case 'git_branch':
          return [GIT_BRANCH_COLUMN_ID, FilterOperator.EQUALS, f.value].join('::');
        case 'git_commit':
          return [GIT_COMMIT_COLUMN_ID, FilterOperator.EQUALS, f.value].join('::');
        default:
          return null;
      }
    })
    .filter((s): s is string => s !== null);
  return result.length > 0 ? result : undefined;
};
