import { useEffect, useMemo } from 'react';

import { pick } from 'lodash';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { loadExperimentViewState, saveExperimentViewState } from '../utils/persistSearchFacets';
import type { ExperimentQueryParamsSearchFacets } from './useExperimentPageSearchFacets';
import { EXPERIMENT_PAGE_QUERY_PARAM_KEYS, useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';

/**
 * Takes care of initializing the search facets from persisted view state and persisting them.
 * Partially replaces GetExperimentRunsContext.
 */
export const usePersistExperimentPageViewState = (
  uiState: ExperimentPageUIState,
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  disabled = false,
) => {
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
        { ...createExperimentPageSearchFacetsState(), ...persistedViewState },
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
