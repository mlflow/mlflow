import { useCallback } from 'react';
import { useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

export const useEvalRunComparisonBaseline = (
  experimentId: string,
  baselineRunUuid: string,
  compareRunUuids: string[],
): ((runUuid: string) => void) => {
  const navigate = useNavigate();
  const locationSearch = useLocation().search;
  return useCallback(
    (nextBaseline: string) => {
      if (nextBaseline === baselineRunUuid || !compareRunUuids.includes(nextBaseline)) return;
      const nextCompare = [...compareRunUuids.filter((uuid) => uuid !== nextBaseline), baselineRunUuid];
      const nextParams = new URLSearchParams(locationSearch);
      nextParams.set('compare', 'true');
      nextParams.set('compareRuns', nextCompare.join(','));
      navigate({
        pathname: Routes.getEvaluationRunDetailsRoute(experimentId, nextBaseline),
        search: `?${nextParams.toString()}`,
      });
    },
    [baselineRunUuid, compareRunUuids, experimentId, locationSearch, navigate],
  );
};
