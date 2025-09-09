import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SearchRunsApiResponse } from '@mlflow/mlflow/src/experiment-tracking/types';
import { MlflowService } from '../../../sdk/MlflowService';
import { useMemo } from 'react';

export const useExperimentEvaluationRunsData = ({
  experimentId,
  enabled,
  filter,
}: {
  experimentId: string;
  enabled: boolean;
  filter: string;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<SearchRunsApiResponse, Error>({
    queryKey: ['SEARCH_RUNS', experimentId, filter],
    queryFn: async () => {
      const requestBody = {
        experiment_ids: [experimentId],
        order_by: ['attributes.start_time DESC'],
        run_view_type: 'ACTIVE_ONLY',
        filter,
      };

      return MlflowService.searchRuns(requestBody);
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  // we determine a run is an eval run if it contains no model outputs
  const evaluationRuns = useMemo(
    () => data?.runs?.filter((run) => (run.outputs?.modelOutputs?.length ?? 0) === 0),
    [data?.runs],
  );

  return {
    data: evaluationRuns,
    refetch,
    isLoading,
    isFetching,
    error,
  };
};
