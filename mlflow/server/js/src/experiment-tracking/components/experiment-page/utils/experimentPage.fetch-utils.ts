import { chunk, isEqual } from 'lodash';
import { AnyAction } from 'redux';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import { MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER } from '../../../../model-registry/constants';
import {
  ATTRIBUTE_COLUMN_SORT_KEY,
  DEFAULT_LIFECYCLE_FILTER,
  DEFAULT_MODEL_VERSION_FILTER,
  DEFAULT_START_TIME,
} from '../../../constants';
import { ViewType } from '../../../sdk/MlflowEnums';
import { KeyValueEntity, LIFECYCLE_FILTER } from '../../../types';
import { EXPERIMENT_LOG_MODEL_HISTORY_TAG } from './experimentPage.common-utils';
import { ThunkDispatch } from '../../../../redux-types';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { RUNS_SEARCH_MAX_RESULTS } from '../../../actions';
import { getUUID } from '../../../../common/utils/ActionUtils';
import { shouldUseRegexpBasedAutoRunsSearchFilter } from '../../../../common/utils/FeatureUtils';

const START_TIME_COLUMN_OFFSET = {
  ALL: null,
  LAST_HOUR: 1 * 60 * 60 * 1000,
  LAST_24_HOURS: 24 * 60 * 60 * 1000,
  LAST_7_DAYS: 7 * 24 * 60 * 60 * 1000,
  LAST_30_DAYS: 30 * 24 * 60 * 60 * 1000,
  LAST_YEAR: 12 * 30 * 24 * 60 * 60 * 1000,
};

const VALID_TABLE_ALIASES = [
  'attribute',
  'attributes',
  'attr',
  'run',
  'metric',
  'metrics',
  'param',
  'params',
  'parameter',
  'tag',
  'tags',
  'dataset',
  'datasets',
];
const SQL_SYNTAX_PATTERN = new RegExp(
  `(${VALID_TABLE_ALIASES.join('|')})\\.\\S+\\s*(>|<|>=|<=|=|!=| like| ilike| rlike| in)`,
  'i',
);

export const RUNS_AUTO_REFRESH_INTERVAL = 30000;

/**
 * This function checks if the sort+model state update has
 * been updated enough and if the change should invoke re-fetching
 * the runs from the back-end. This enables differentiation between
 * front-end and back-end filtering.
 */
export const shouldRefetchRuns = (
  currentSearchFacetsState: ExperimentPageSearchFacetsState,
  newSearchFacetsState: ExperimentPageSearchFacetsState,
) =>
  !isEqual(currentSearchFacetsState.searchFilter, newSearchFacetsState.searchFilter) ||
  !isEqual(currentSearchFacetsState.orderByAsc, newSearchFacetsState.orderByAsc) ||
  !isEqual(currentSearchFacetsState.orderByKey, newSearchFacetsState.orderByKey) ||
  !isEqual(currentSearchFacetsState.lifecycleFilter, newSearchFacetsState.lifecycleFilter) ||
  !isEqual(currentSearchFacetsState.startTime, newSearchFacetsState.startTime) ||
  !isEqual(currentSearchFacetsState.datasetsFilter, newSearchFacetsState.datasetsFilter);

/**
 * Creates "order by" SQL expression
 */
const createOrderByExpression = ({ orderByKey, orderByAsc }: ExperimentPageSearchFacetsState) => {
  if (orderByKey) {
    return orderByAsc ? [orderByKey + ' ASC'] : [orderByKey + ' DESC'];
  }
  return [];
};

/**
 * Creates SQL expression for filtering by run start time
 */
const createStartTimeExpression = ({ startTime }: ExperimentPageSearchFacetsState, referenceTime: number) => {
  const offset = START_TIME_COLUMN_OFFSET[startTime as keyof typeof START_TIME_COLUMN_OFFSET];
  if (!startTime || !offset || startTime === 'ALL') {
    return null;
  }
  const startTimeOffset = referenceTime - offset;

  return `attributes.start_time >= ${startTimeOffset}`;
};

/**
 * Creates SQL expression for filtering by selected datasets
 */
const createDatasetsFilterExpression = ({ datasetsFilter }: ExperimentPageSearchFacetsState) => {
  if (datasetsFilter.length === 0) {
    return null;
  }
  const datasetNames = datasetsFilter.map((dataset) => `'${dataset.name}'`).join(',');
  const datasetDigests = datasetsFilter.map((dataset) => `'${dataset.digest}'`).join(',');

  return `dataset.name IN (${datasetNames}) AND dataset.digest IN (${datasetDigests})`;
};

export const detectSqlSyntaxInSearchQuery = (searchFilter: string) => {
  return SQL_SYNTAX_PATTERN.test(searchFilter);
};

export const createQuickRegexpSearchFilter = (searchFilter: string) =>
  `attributes.run_name RLIKE '${searchFilter.replace(/'/g, "\\'")}'`;

/**
 * Combines search filter and start time SQL expressions
 */
const createFilterExpression = (
  { searchFilter }: ExperimentPageSearchFacetsState,
  startTimeExpression: string | null,
  datasetsFilterExpression: string | null,
) => {
  if (
    shouldUseRegexpBasedAutoRunsSearchFilter() &&
    searchFilter.length > 0 &&
    !detectSqlSyntaxInSearchQuery(searchFilter)
  ) {
    return createQuickRegexpSearchFilter(searchFilter);
  }

  const activeFilters = [];
  if (searchFilter) activeFilters.push(searchFilter);
  if (startTimeExpression) activeFilters.push(startTimeExpression);
  if (datasetsFilterExpression) activeFilters.push(datasetsFilterExpression);

  if (activeFilters.length === 0) return undefined;
  return activeFilters.join(' and ');
};

/**
 * If this function returns true, the ExperimentView should nest children underneath their parents
 * and fetch all root level parents of visible runs. If this function returns false, the views will
 * not nest children or fetch any additional parents. Will always return true if the orderByKey is
 * 'attributes.start_time'
 */
const shouldNestChildrenAndFetchParents = ({ orderByKey, searchFilter }: ExperimentPageSearchFacetsState) =>
  (!orderByKey && !searchFilter) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE;

/**
 *
 * Function creates API-compatible query object basing on the given criteria.
 * @param experimentIds IDs of experiments to be queries for runs
 * @param searchFacetsState the sort/filter model to use
 * @param referenceTime reference time to calculate startTime filter
 * @param pageToken next page token if fetching the next page
 */
export const createSearchRunsParams = (
  experimentIds: string[],
  searchFacetsState: ExperimentPageSearchFacetsState & { runsPinned: string[] },
  referenceTime: number,
  pageToken?: string,
  maxResults?: number,
) => {
  const runViewType =
    searchFacetsState.lifecycleFilter === LIFECYCLE_FILTER.ACTIVE ? ViewType.ACTIVE_ONLY : ViewType.DELETED_ONLY;

  const { runsPinned = undefined } = searchFacetsState;

  const orderBy = createOrderByExpression(searchFacetsState);
  const startTimeExpression = createStartTimeExpression(searchFacetsState, referenceTime);
  const datasetsFilterExpression = createDatasetsFilterExpression(searchFacetsState);
  const filter = createFilterExpression(searchFacetsState, startTimeExpression, datasetsFilterExpression);
  const shouldFetchParents = shouldNestChildrenAndFetchParents(searchFacetsState);

  return {
    // Experiment IDs
    experimentIds,

    // Filters and sort options
    filter,
    runViewType,
    orderBy,
    shouldFetchParents,

    // Next page token for loading more runs
    pageToken,
    runsPinned,

    maxResults: maxResults || RUNS_SEARCH_MAX_RESULTS,
  };
};
/**
 * Function checks if given runs set contain info about log model history and if true,
 * fetches model versions for them
 *
 * @param runsPayload runs payload returned from the searchRuns API
 * @param actionCreator redux-thunk action creator that for search model versions action
 * @param dispatch redux-compatible dispatch function
 */
export const fetchModelVersionsForRuns = (
  runsPayload: {
    info: {
      run_id: string;
    };
    data: {
      tags: KeyValueEntity[];
    };
  }[],
  actionCreator: typeof searchModelVersionsApi,
  dispatch: ThunkDispatch,
) => {
  const runsWithLogModelHistory = runsPayload.filter((run) =>
    run.data.tags.some((t) => t.key === EXPERIMENT_LOG_MODEL_HISTORY_TAG),
  );

  chunk(runsWithLogModelHistory, MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER).forEach((runsChunk) => {
    // eslint-disable-next-line prefer-const
    let maxResults = undefined;
    const action = actionCreator(
      {
        run_id: runsChunk.map((run) => run.info.run_id),
      },
      getUUID(),
      maxResults,
    );
    dispatch(action);
  });
};

/**
 * Function consumes a search state facets object and returns `true`
 * if at least one filter-related facet is not-default meaning that runs
 * are currently filtered.
 */
export const isSearchFacetsFilterUsed = (currentSearchFacetsState: ExperimentPageSearchFacetsState) => {
  const { lifecycleFilter, modelVersionFilter, datasetsFilter, searchFilter, startTime } = currentSearchFacetsState;
  return Boolean(
    lifecycleFilter !== DEFAULT_LIFECYCLE_FILTER ||
      modelVersionFilter !== DEFAULT_MODEL_VERSION_FILTER ||
      datasetsFilter.length !== 0 ||
      searchFilter ||
      startTime !== DEFAULT_START_TIME,
  );
};
