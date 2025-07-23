import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { isEmpty } from 'lodash';
import invariant from 'invariant';
import { useMemo } from 'react';

const QUERY_KEY = 'EXPERIMENT_CONTAINS_TRAINING_RUNS';

/**
 * Hook for checking if there are any runs in a given experiment.
 * Returns `containsRuns` set to `true` if there's at least one run, `false` otherwise.
 */
export const useExperimentContainsTrainingRuns = ({
  experimentId,
  enabled,
}: {
  experimentId?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, error } = useQuery(
    [QUERY_KEY, experimentId],
    async () => {
      invariant(experimentId, 'experimentId is required');
      const experimentIds = [experimentId];

      const ret = await MlflowService.searchRuns({
        experiment_ids: experimentIds,
        max_results: 1,
      });

      return ret;
    },
    {
      enabled: Boolean(experimentId) && enabled,
    },
  );

  const containsRuns = useMemo(() => !isLoading && !isEmpty(data?.runs), [isLoading, data]);

  return {
    containsRuns,
    isLoading: isLoading && enabled && Boolean(experimentId),
  };
};
