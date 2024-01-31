import { useContext } from 'react';
import { GetExperimentRunsContext, GetExperimentRunsContextType } from '../contexts/GetExperimentRunsContext';
import { shouldEnableShareExperimentViewByTags } from '../../../../common/utils/FeatureUtils';
import { createExperimentPageSearchFacetsStateV2 } from '../models/ExperimentPageSearchFacetsStateV2';
import { createExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';
import { loadMoreRunsApi, searchRunsApi, searchRunsPayload } from '../../../actions';
import { searchModelVersionsApi } from '../../../../model-registry/actions';

// Empty/noop context value, is returned when the hook is disabled
const emptyContextValue: GetExperimentRunsContextType = {
  actions: {
    searchRunsApi,
    loadMoreRunsApi,
    searchRunsPayload,
    searchModelVersionsApi,
  },
  fetchExperimentRuns: () => {},
  isPristine: () => false,
  isLoadingRuns: false,
  loadMoreRuns: async () => [],
  moreRunsAvailable: false,
  refreshRuns: () => {},
  searchFacetsState: {
    ...createExperimentPageSearchFacetsStateV2(),
    ...createExperimentPageUIStateV2(),
    compareRunsMode: undefined,
  },
  requestError: null,
  updateSearchFacets: () => {},
};

/**
 * Legacy experiment page view state retrieval hook using context for storing the facets/UI data.
 * TODO: remove after migrating to "useExperimentPageRuns"
 */
export const useFetchExperimentRuns = () => {
  // The eslint rule can be disabled safelyt, the condition based on feature flag evaluation is stable
  /* eslint-disable react-hooks/rules-of-hooks */
  if (shouldEnableShareExperimentViewByTags()) {
    // If the feature flag is enabled, the hook should not be used anymore and returns empty/noop values
    console.warn(
      '[useFetchExperimentRuns] Invalid hook usage: useFetchExperimentRuns is not supported anymore when shareExperimentViewByTags flag is enabled.',
    );
    return emptyContextValue;
  }
  const getExperimentRunsContextValue = useContext(GetExperimentRunsContext);

  if (!getExperimentRunsContextValue) {
    throw new Error('Trying to use SearchExperimentRunsContext actions outside of the context!');
  }

  return getExperimentRunsContextValue;
};
