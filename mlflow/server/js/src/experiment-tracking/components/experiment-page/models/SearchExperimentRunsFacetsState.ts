import {
  ATTRIBUTE_COLUMN_LABELS,
  COLUMN_TYPES,
  DEFAULT_LIFECYCLE_FILTER,
  DEFAULT_MODEL_VERSION_FILTER,
  DEFAULT_ORDER_BY_ASC,
  DEFAULT_ORDER_BY_KEY,
  DEFAULT_START_TIME,
} from '../../../constants';
import { makeCanonicalSortKey } from '../utils/experimentPage.column-utils';

const DEFAULT_SELECTED_COLUMNS = [
  // "Source" and "Model" columns are visible by default
  makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.SOURCE),
  makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.MODELS),
];

/**
 * Defines persistable model respresenting sort and filter values
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
   * Filter of model versions to display
   */
  modelVersionFilter = DEFAULT_MODEL_VERSION_FILTER;

  /**
   * Currently selected columns
   */
  selectedColumns: string[] = [...DEFAULT_SELECTED_COLUMNS];

  /**
   * Object mapping run UUIDs (strings) to booleans, where a boolean value of true indicates that
   * a run has been expanded (its child runs are visible).
   */
  runsExpanded: Record<string, boolean> = {};

  /**
   * List of pinned row UUIDs
   */
  runsPinned: string[] = [];
}
