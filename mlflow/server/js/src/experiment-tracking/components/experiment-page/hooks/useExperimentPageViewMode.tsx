import { useNavigate, useSearchParams } from '../../../../common/utils/RoutingUtils';
import { ExperimentPageTabName } from '../../../constants';
import Routes from '../../../routes';
import { type ExperimentViewRunsCompareMode } from '../../../types';

export const EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY = 'compareRunsMode';

export const getExperimentPageDefaultViewMode = (): ExperimentViewRunsCompareMode => 'TABLE';

// This map is being used to wire routes to certain view modes
const viewModeToRouteMap: Partial<Record<ExperimentViewRunsCompareMode, (experimentId: string) => void>> = {
  MODELS: (experimentId: string) => Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models),
};
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
    // Check if the new mode should actually navigate to a different route instead of just changing the query param
    if (newCompareRunsMode in viewModeToRouteMap && experimentId) {
      const route = viewModeToRouteMap[newCompareRunsMode]?.(experimentId);
      if (route) {
        navigate(route);
        return;
      }
    }
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
