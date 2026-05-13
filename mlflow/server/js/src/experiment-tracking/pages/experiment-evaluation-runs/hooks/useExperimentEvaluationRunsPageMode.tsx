import { coerceToEnum } from '@databricks/web-shared/utils';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

export enum ExperimentEvaluationRunsPageMode {
  TRACES = 'traces',
  CHARTS = 'charts',
}

const MODE_SEARCH_KEY = 'viewMode';

export const useExperimentEvaluationRunsPageMode = () => {
  const [params, setParams] = useSearchParams();
  const viewMode = coerceToEnum(
    ExperimentEvaluationRunsPageMode,
    params.get(MODE_SEARCH_KEY),
    ExperimentEvaluationRunsPageMode.TRACES,
  );

  const setViewMode = (newMode: ExperimentEvaluationRunsPageMode) => {
    setParams((prevParams) => {
      const newParams = new URLSearchParams(prevParams);
      newParams.set(MODE_SEARCH_KEY, newMode);
      return newParams;
    });
  };

  return {
    viewMode,
    setViewMode,
  };
};
