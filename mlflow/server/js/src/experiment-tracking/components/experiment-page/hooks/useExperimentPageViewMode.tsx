import { shouldUseExperimentPageChartViewAsDefault } from '../../../../common/utils/FeatureUtils';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';
import { type ExperimentViewRunsCompareMode } from '../../../types';

const EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY = 'compareRunsMode';

export const getExperimentPageDefaultViewMode = (): ExperimentViewRunsCompareMode =>
  shouldUseExperimentPageChartViewAsDefault() ? 'CHART' : 'TABLE';

/**
 * Hook using search params to retrieve and update the current experiment page runs view mode.
 */
export const useExperimentPageViewMode = (): [
  ExperimentViewRunsCompareMode,
  (newCompareRunsMode: ExperimentViewRunsCompareMode) => void,
] => {
  const [params, setParams] = useSearchParams();

  const mode =
    (params.get(EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY) as ExperimentViewRunsCompareMode) ||
    getExperimentPageDefaultViewMode();
  const setMode = (newCompareRunsMode: ExperimentViewRunsCompareMode) => {
    setParams(
      (currentParams) => {
        currentParams.set(EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY, newCompareRunsMode || '');
        return currentParams;
      },
      { replace: false },
    );
  };

  return [mode, setMode];
};
