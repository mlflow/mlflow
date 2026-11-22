import { useCallback } from 'react';
import { useNavigate, useSearchParamSelector } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { withoutSpan } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4SearchParams';

export const useEvalRunComparisonBaseline = (
  experimentId: string,
  baselineRunUuid: string,
  compareRunUuids: string[],
): ((runUuid: string) => void) => {
  const navigate = useNavigate();
  const [locationSearch] = useSearchParamSelector(withoutSpan);
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
