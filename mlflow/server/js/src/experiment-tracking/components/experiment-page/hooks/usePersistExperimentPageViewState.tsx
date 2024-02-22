import { useEffect, useMemo } from 'react';

import { pick } from 'lodash';
import { shouldEnableShareExperimentViewByTags } from '../../../../common/utils/FeatureUtils';
import { createExperimentPageSearchFacetsStateV2 } from '../models/ExperimentPageSearchFacetsStateV2';
import { ExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';
import { loadExperimentViewState, saveExperimentViewState } from '../utils/persistSearchFacets';
import {
  EXPERIMENT_PAGE_QUERY_PARAM_KEYS,
  ExperimentQueryParamsSearchFacets,
  useUpdateExperimentPageSearchFacets,
} from './useExperimentPageSearchFacets';

/**
 * Takes care of initializing the search facets from persisted view state and persisting them.
 * Partially replaces GetExperimentRunsContext.
 */
export const usePersistExperimentPageViewState = (
  uiState: ExperimentPageUIStateV2,
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  disabled = false,
) => {
  // We can disable this eslint rule because condition uses a stable feature flag evaluation
  /* eslint-disable react-hooks/rules-of-hooks */
  if (!shouldEnableShareExperimentViewByTags()) {
    // Don't use the new API if the feature flag is disabled
    return;
  }

  const setSearchFacets = useUpdateExperimentPageSearchFacets();

  const persistKey = useMemo(() => (experimentIds ? JSON.stringify(experimentIds.sort()) : null), [experimentIds]);

  // If there are no query params visible in the address bar, either reinstantiate
  // them from persisted view state or use default values.
  useEffect(() => {
    if (disabled) {
      return;
    }
    if (!searchFacets) {
      const persistedViewState = persistKey ? loadExperimentViewState(persistKey) : null;
      const rebuiltViewState = pick(
        { ...createExperimentPageSearchFacetsStateV2(), ...persistedViewState },
        EXPERIMENT_PAGE_QUERY_PARAM_KEYS,
      );
      setSearchFacets(rebuiltViewState, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchFacets, persistKey, disabled]);

  // Persist complete view state in local storage when either search facets or UI state change
  useEffect(() => {
    if (!searchFacets || !persistKey || disabled) {
      return;
    }
    saveExperimentViewState({ ...searchFacets, ...uiState }, persistKey);
  }, [searchFacets, uiState, persistKey, disabled]);
};
