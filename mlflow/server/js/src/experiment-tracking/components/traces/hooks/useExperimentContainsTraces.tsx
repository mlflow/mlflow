import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { isEmpty } from 'lodash';
import invariant from 'invariant';

const QUERY_KEY = 'EXPERIMENT_CONTAINS_TRACES';

/**
 * Hook for checking if there are any traces for a given experiment.
 * Returns `containsTraces` set to `true` if there's at least one trace, `false` otherwise.
 */
export const useExperimentContainsTraces = ({
  experimentId,
  enabled,
}: {
  experimentId?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading } = useQuery(
    [QUERY_KEY, experimentId],
    async () => {
      invariant(experimentId, 'experimentId is required');
      const experimentIds = [experimentId];

      return MlflowService.getExperimentTraces(experimentIds, 'timestamp_ms DESC', undefined, undefined, 1);
    },
    {
      enabled: enabled && Boolean(experimentId),
    },
  );

  const containsTraces = !isLoading && !isEmpty(data?.traces);
  return {
    containsTraces,
    isLoading: isLoading && enabled && Boolean(experimentId),
  };
};
