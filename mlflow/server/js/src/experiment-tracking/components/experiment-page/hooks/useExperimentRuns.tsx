import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useDispatch } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { loadMoreRunsApi, searchRunsApi } from '../../../actions';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { createSearchRunsParams, fetchModelVersionsForRuns } from '../utils/experimentPage.fetch-utils';
import type { ExperimentRunsSelectorResult } from '../utils/experimentRuns.selector';
import { experimentRunsSelector } from '../utils/experimentRuns.selector';
import type { ExperimentQueryParamsSearchFacets } from './useExperimentPageSearchFacets';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import { PredefinedError } from '@databricks/web-shared/errors';
import { mapErrorWrapperToPredefinedError } from '../../../../common/utils/ErrorUtils';
import { shouldUsePredefinedErrorsInExperimentTracking } from '../../../../common/utils/FeatureUtils';
import Utils from '../../../../common/utils/Utils';
import { useExperimentRunsAutoRefresh } from './useExperimentRunsAutoRefresh';
import type { RunEntity, SearchRunsApiResponse } from '../../../types';

export type FetchRunsHookParams = ReturnType<typeof createSearchRunsParams> & {
  requestedFacets: ExperimentPageSearchFacetsState;
};

export type FetchRunsHookFunction = (
  params: FetchRunsHookParams,
  options?: {
    isAutoRefreshing?: boolean;
    discardResultsFn?: (lastRequestedParams: FetchRunsHookParams, response?: SearchRunsApiResponse) => boolean;
  },
) => Promise<{ runs: RunEntity[]; next_page_token?: string }>;

// Calculate actual params to use for fetching runs
const createFetchRunsRequestParams = (
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  runsPinned: string[],
): FetchRunsHookParams | null => {
  if (!searchFacets || !experimentIds.length) {
    return null;
  }
  const searchParams = createSearchRunsParams(experimentIds, { ...searchFacets, runsPinned }, Date.now());
  return { ...searchParams, requestedFacets: searchFacets };
};

/**
 * This hook will request for new runs data based on the current experiment page search facets and UI state.
 * Replaces GetExperimentRunsContext and a substantial portion of <ExperimentRuns> component stack.
 */
export const useExperimentRuns = (
  uiState: ExperimentPageUIState,
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  disabled = false,
) => {
  const dispatch = useDispatch<ThunkDispatch>();

  const [runsData, setRunsData] = useState<ExperimentRunsSelectorResult>(() => createEmptyRunsResult());

  const enableWorkspaceModelsRegistryCall = true;

  const persistKey = useMemo(() => (experimentIds ? JSON.stringify(experimentIds.sort()) : null), [experimentIds]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);
  const [isInitialLoadingRuns, setIsInitialLoadingRuns] = useState(true);
  const [nextPageToken, setNextPageToken] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<ErrorWrapper | Error | null>(null);
  const cachedPinnedRuns = useRef<string[]>([]);

  const lastFetchedTime = useRef<number | null>(null);
  const lastRequestedParams = useRef<FetchRunsHookParams | null>(null);
  const lastSuccessfulRequestedParams = useRef<FetchRunsHookParams | null>(null);

  // Reset initial loading state when experiment IDs change
  useEffect(() => {
    if (disabled) {
      return;
    }
    setIsInitialLoadingRuns(true);
    setRunsData(createEmptyRunsResult());
  }, [persistKey, disabled]);

  const setResultRunsData = useCallback(
    (store: ReduxState, experimentIds: string[], requestedFacets: ExperimentPageSearchFacetsState) => {
      setRunsData(
        experimentRunsSelector(store, {
          datasetsFilter: requestedFacets.datasetsFilter,
          lifecycleFilter: requestedFacets.lifecycleFilter,
          modelVersionFilter: requestedFacets.modelVersionFilter,
          // In the new version of the view state, experiment IDs are used instead of full experiment entities:
          experiments: [],
          experimentIds,
        }),
      );
    },
    [],
  );

  useEffect(() => {
    cachedPinnedRuns.current = uiState.runsPinned;
  }, [uiState.runsPinned]);

  const loadModelVersions = useCallback(
    (runs: Parameters<typeof fetchModelVersionsForRuns>[0]) => {
      const handleModelVersionLoadFailure = (error: Error | ErrorWrapper) => {
        const normalizedError =
          (error instanceof ErrorWrapper ? mapErrorWrapperToPredefinedError(error) : error) ?? error;
        const message =
          normalizedError instanceof ErrorWrapper ? normalizedError.getMessageField() : normalizedError.message;
        Utils.displayGlobalErrorNotification(`Failed to load model versions for runs: ${message}`);
      };
      if (enableWorkspaceModelsRegistryCall) {
        fetchModelVersionsForRuns(runs || [], searchModelVersionsApi, dispatch).catch(handleModelVersionLoadFailure);
      }
    },
    [dispatch, enableWorkspaceModelsRegistryCall],
  );

  // Main function for fetching runs
  const fetchRuns: FetchRunsHookFunction = useCallback(
    (fetchParams, options = {}) =>
      dispatch((thunkDispatch: ThunkDispatch, getStore: () => ReduxState) => {
        // If we're auto-refreshing, we don't want to show the loading spinner and
        // we don't want to update the last requested params - they're used to determine
        // whether to discard results when the automatically fetched data changes.
        if (!options.isAutoRefreshing) {
          setIsLoadingRuns(true);
          lastRequestedParams.current = fetchParams;
        }
        return thunkDispatch((fetchParams.pageToken ? loadMoreRunsApi : searchRunsApi)(fetchParams))
          .then(async ({ value }) => {
            lastFetchedTime.current = Date.now();

            setIsLoadingRuns(false);
            setIsInitialLoadingRuns(false);
            setRequestError(null);

            if (lastRequestedParams.current && options.discardResultsFn?.(lastRequestedParams.current, value)) {
              return value;
            }

            lastSuccessfulRequestedParams.current = fetchParams;
            setNextPageToken(value.next_page_token || null);

            // We rely on redux reducer to update the state with new runs data,
            // then we pick it up from the store. This benefits other pages that use same data
            // from the same store slice (e.g. run details page). Will be changed when moving to graphQL.
            setResultRunsData(getStore(), fetchParams.experimentIds, fetchParams.requestedFacets);

            // In the end, load model versions for the fetched runs
            loadModelVersions(value.runs || []);
            return value;
          })
          .catch((e: ErrorWrapper | PredefinedError) => {
            setIsLoadingRuns(false);
            setIsInitialLoadingRuns(false);
            if (shouldUsePredefinedErrorsInExperimentTracking()) {
              // If it's already a PredefinedError, we don't need to map it again
              if (e instanceof PredefinedError) {
                setRequestError(e);
                return;
              }
              const maybePredefinedError = mapErrorWrapperToPredefinedError(e);
              if (maybePredefinedError) {
                setRequestError(maybePredefinedError);
                return;
              }
            }
            setRequestError(e);
            if (!shouldUsePredefinedErrorsInExperimentTracking()) {
              Utils.logErrorAndNotifyUser(e);
            }
          });
      }),
    [dispatch, setResultRunsData, loadModelVersions],
  );

  // Fetch runs when new request params are available
  // (e.g. after search facets change)
  useEffect(() => {
    if (disabled) {
      return;
    }
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds, cachedPinnedRuns.current);
    if (requestParams) {
      fetchRuns(requestParams);
    }
  }, [fetchRuns, dispatch, disabled, searchFacets, experimentIds]);

  const loadMoreRuns = async () => {
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds, cachedPinnedRuns.current);
    if (!nextPageToken || !requestParams) {
      return [];
    }
    return fetchRuns({ ...requestParams, pageToken: nextPageToken });
  };

  const refreshRuns = useCallback(() => {
    if (lastSuccessfulRequestedParams.current) {
      fetchRuns({ ...lastSuccessfulRequestedParams.current, pageToken: undefined });
    }
  }, [fetchRuns]);

  useExperimentRunsAutoRefresh({
    experimentIds,
    fetchRuns,
    searchFacets,
    enabled: uiState.autoRefreshEnabled,
    cachedPinnedRuns,
    runsData,
    isLoadingRuns: isLoadingRuns,
    lastFetchedTime,
  });

  return {
    isLoadingRuns,
    moreRunsAvailable: Boolean(nextPageToken),
    refreshRuns,
    loadMoreRuns,
    isInitialLoadingRuns,
    runsData,
    requestError,
  };
};

const createEmptyRunsResult = () => ({
  datasetsList: [],
  experimentTags: {},
  metricKeyList: [],
  metricsList: [],
  modelVersionsByRunUuid: {},
  paramKeyList: [],
  paramsList: [],
  runInfos: [],
  runUuidsMatchingFilter: [],
  tagsList: [],
  inputsOutputsList: [],
});
