import { useEffect, useRef } from 'react';
import { UpdateExperimentSearchFacetsFn } from '../../../types';

export const useChartViewByDefault = (
  isLoadingRuns: boolean,
  metricKeyList: string[],
  updateSearchFacets: UpdateExperimentSearchFacetsFn,
) => {
  const hasAutomaticallyChangedModeAlready = useRef(false);
  // Default to chart view if there are any metrics in the runsData.
  useEffect(() => {
    // - Let's wait until the runs are finished loading.
    // - If the mode was automatically changed already, do nothing.
    if (isLoadingRuns || hasAutomaticallyChangedModeAlready.current) {
      return;
    }
    if (metricKeyList.length > 0) {
      updateSearchFacets({ compareRunsMode: 'CHART' });
      hasAutomaticallyChangedModeAlready.current = true;
    }
  }, [isLoadingRuns, metricKeyList.length, updateSearchFacets]);
};
