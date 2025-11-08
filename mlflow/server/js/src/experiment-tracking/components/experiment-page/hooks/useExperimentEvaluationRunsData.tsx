import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { SearchRunsApiResponse } from '@mlflow/mlflow/src/experiment-tracking/types';
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

  const { evaluationRuns, trainingRuns } = useMemo(() => {
    if (!data?.runs) {
      return { evaluationRuns: [], trainingRuns: [] };
    }
    return data.runs.reduce(
      (acc, run) => {
        const isTrainingRun = run.outputs?.modelOutputs?.length ?? 0;

        if (isTrainingRun) {
          acc.trainingRuns.push(run);
        } else {
          acc.evaluationRuns.push(run);
        }

        return acc;
      },
      { evaluationRuns: [], trainingRuns: [] } as { evaluationRuns: typeof data.runs; trainingRuns: typeof data.runs },
    );
  }, [data]);

  return {
    data: evaluationRuns,
    trainingRuns,
    refetch,
    isLoading,
    isFetching,
    error,
  };
};
