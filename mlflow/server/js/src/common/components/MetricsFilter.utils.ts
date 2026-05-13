import { FilterOperator, USER_COLUMN_ID } from '@databricks/web-shared/genai-traces-table';
import { MLFLOW_TRACE_USER_KEY, createTraceMetadataFilter } from '@databricks/web-shared/model-trace-explorer';

/**
 * Curated list of columns the metrics API can filter on with `=`.
 * Add a new entry here AND in `translateToMetricsFilters` and
 * `translateToTracesPageFilters` to expose more dimensions.
 */
export type MetricFilterColumn = 'user';

export interface MetricFilter {
  column: MetricFilterColumn;
  value: string;
}

export interface MetricFilterColumnOption {
  value: MetricFilterColumn;
  label: string;
}

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
        default:
          return null;
      }
    })
    .filter((s): s is string => s !== null);
  return result.length > 0 ? result : undefined;
};
