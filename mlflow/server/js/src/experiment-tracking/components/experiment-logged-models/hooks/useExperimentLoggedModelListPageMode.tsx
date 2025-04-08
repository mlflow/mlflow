import { coerceToEnum } from '@databricks/web-shared/utils';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

export enum ExperimentLoggedModelListPageMode {
  TABLE = 'TABLE',
  CHART = 'CHART',
}

const VIEW_MODE_QUERY_PARAM = 'viewMode';

export const useExperimentLoggedModelListPageMode = () => {
  const [params, setParams] = useSearchParams();
  const viewMode = coerceToEnum(
    ExperimentLoggedModelListPageMode,
    params.get(VIEW_MODE_QUERY_PARAM),
    ExperimentLoggedModelListPageMode.TABLE,
  );
  const setViewMode = (mode: ExperimentLoggedModelListPageMode) => {
    setParams({ [VIEW_MODE_QUERY_PARAM]: mode });
  };
  return { viewMode, setViewMode } as const;
};
