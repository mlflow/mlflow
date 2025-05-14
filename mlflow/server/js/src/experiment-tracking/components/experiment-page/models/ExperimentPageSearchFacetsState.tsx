import {
  DEFAULT_LIFECYCLE_FILTER,
  DEFAULT_MODEL_VERSION_FILTER,
  DEFAULT_ORDER_BY_ASC,
  DEFAULT_ORDER_BY_KEY,
  DEFAULT_START_TIME,
} from '../../../constants';
import type { DatasetSummary, LIFECYCLE_FILTER, MODEL_VERSION_FILTER } from '../../../types';

/**
 * Defines persistable model representing sort and filter values
 * used by runs table and controls
 */
export interface ExperimentPageSearchFacetsState {
  /**
   * SQL-like query string used to filter runs, e.g. "params.alpha = '0.5'"
   */
  searchFilter: string;

  /**
   * Canonical order_by key like "params.`alpha`". May be null to indicate the table
   * should use the natural row ordering provided by the server.
   */
  orderByKey: string;

  /**
   * Whether the order imposed by orderByKey should be ascending or descending.
   */
  orderByAsc: boolean;

  /**
   * Filter key to show results based on start time
   */
  startTime: string;

  /**
   * Lifecycle filter of runs to display
   */
  lifecycleFilter: LIFECYCLE_FILTER;

  /**
   * Datasets filter of runs to display
   */
  datasetsFilter: DatasetSummary[];

  /**
   * Filter of model versions to display
   */
  modelVersionFilter: MODEL_VERSION_FILTER;
}

/**
 * Defines default experiment page search facets state.
 */
export const createExperimentPageSearchFacetsState = (): ExperimentPageSearchFacetsState => ({
  searchFilter: '',
  orderByKey: DEFAULT_ORDER_BY_KEY,
  orderByAsc: DEFAULT_ORDER_BY_ASC,
  startTime: DEFAULT_START_TIME,
  lifecycleFilter: DEFAULT_LIFECYCLE_FILTER,
  datasetsFilter: [],
  modelVersionFilter: DEFAULT_MODEL_VERSION_FILTER,
});
