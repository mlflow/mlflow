import { shouldUseExperimentPageChartViewAsDefault } from '../../../../common/utils/FeatureUtils';
import { useNavigate, useSearchParams } from '../../../../common/utils/RoutingUtils';
import { ExperimentPageTabName } from '../../../constants';
import Routes from '../../../routes';
import { type ExperimentViewRunsCompareMode } from '../../../types';

export const EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY = 'compareRunsMode';

export const getExperimentPageDefaultViewMode = (): ExperimentViewRunsCompareMode =>
  shouldUseExperimentPageChartViewAsDefault() ? 'CHART' : 'TABLE';

/**
 * Hook using search params to retrieve and update the current experiment page runs view mode.
 * Handles legacy part of the mode switching, based on "compareRunsMode" query parameter.
 * Modern part of the mode switching is handled by <ExperimentViewRunsModeSwitchV2> which works using route params.
 */
export const useExperimentPageViewMode = (): [
  ExperimentViewRunsCompareMode,
  (newCompareRunsMode: ExperimentViewRunsCompareMode, experimentId?: string) => void,
] => {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();

  const mode =
    (params.get(EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY) as ExperimentViewRunsCompareMode) ||
    getExperimentPageDefaultViewMode();
  const setMode = (newCompareRunsMode: ExperimentViewRunsCompareMode, experimentId?: string) => {
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
