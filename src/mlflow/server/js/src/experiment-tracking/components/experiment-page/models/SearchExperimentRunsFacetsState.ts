import {
  ATTRIBUTE_COLUMN_LABELS,
  COLUMN_TYPES,
  DEFAULT_LIFECYCLE_FILTER,
  DEFAULT_MODEL_VERSION_FILTER,
  DEFAULT_ORDER_BY_ASC,
  DEFAULT_ORDER_BY_KEY,
  DEFAULT_START_TIME,
} from '../../../constants';
import { SerializedRunsChartsCardConfigCard } from '../../runs-charts/runs-charts.types';
import { makeCanonicalSortKey } from '../utils/experimentPage.common-utils';
import type { DatasetSummary, ExperimentViewRunsCompareMode, ChartSectionConfig } from '../../../types';

const getDefaultSelectedColumns = () => {
  const result = [
    // "Source" and "Model" columns are visible by default
    makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.SOURCE),
    makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.MODELS),
    makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATASET),
  ];

  return result;
};

/**
 * Function consumes a search state facets object and returns one
 * with cleared filter-related fields while leaving
 * selected columns, chart state etc.
 */
export const clearSearchExperimentsFacetsFilters = (currentSearchFacetsState: SearchExperimentRunsFacetsState) => {
  const { lifecycleFilter, datasetsFilter, modelVersionFilter, searchFilter, startTime, orderByAsc, orderByKey } =
    new SearchExperimentRunsFacetsState();
  return {
    ...currentSearchFacetsState,
    lifecycleFilter,
    datasetsFilter,
    modelVersionFilter,
    searchFilter,
    startTime,
    orderByAsc,
    orderByKey,
  };
};

/**
 * Function consumes a search state facets object and returns `true`
 * if at least one filter-related facet is not-default meaning that runs
 * are currently filtered.
 */
export const isSearchFacetsFilterUsed = (currentSearchFacetsState: SearchExperimentRunsFacetsState) => {
  const { lifecycleFilter, modelVersionFilter, datasetsFilter, searchFilter, startTime } = currentSearchFacetsState;
  return Boolean(
    lifecycleFilter !== DEFAULT_LIFECYCLE_FILTER ||
      modelVersionFilter !== DEFAULT_MODEL_VERSION_FILTER ||
      datasetsFilter.length !== 0 ||
      searchFilter ||
      startTime !== DEFAULT_START_TIME,
  );
};

/**
 * Defines persistable model representing sort and filter values
 * used by runs table and controls
 */
export class SearchExperimentRunsFacetsState {
  /**
   * SQL-like query string used to filter runs, e.g. "params.alpha = '0.5'"
   */
  searchFilter = '';

  /**
   * Canonical order_by key like "params.`alpha`". May be null to indicate the table
   * should use the natural row ordering provided by the server.
   */
  orderByKey = DEFAULT_ORDER_BY_KEY;

  /**
   * Whether the order imposed by orderByKey should be ascending or descending.
   */
  orderByAsc = DEFAULT_ORDER_BY_ASC;

  /**
   * Filter key to show results based on start time
   */
  startTime = DEFAULT_START_TIME;

  /**
   * Lifecycle filter of runs to display
   */
  lifecycleFilter = DEFAULT_LIFECYCLE_FILTER;

  /**
   * Datasets filter of runs to display
   */
  datasetsFilter: DatasetSummary[] = [];

  /**
   * Filter of model versions to display
   */
  modelVersionFilter = DEFAULT_MODEL_VERSION_FILTER;

  /**
   * Currently selected columns
   */
  selectedColumns: string[] = getDefaultSelectedColumns();

  /**
   * Object mapping run UUIDs (strings) to booleans, where a boolean value of true indicates that
   * a run has been expanded (its child runs are visible).
   */
  runsExpanded: Record<string, boolean> = {};

  /**
   * List of pinned row UUIDs
   */
  runsPinned: string[] = [];

  /**
   * List of hidden row UUIDs
   */
  runsHidden: string[] = [];

  /**
   * Current run comparison mode (either chart or artifact).
   * If set to "undefined", the table view should be displayed.
   */
  compareRunsMode: ExperimentViewRunsCompareMode = undefined;

  /**
   * Currently configured charts for comparing runs, if any.
   */
  compareRunCharts?: SerializedRunsChartsCardConfigCard[];

  /**
   * Sections for grouping compare runs charts
   */
  compareRunSections?: ChartSectionConfig[];
}
