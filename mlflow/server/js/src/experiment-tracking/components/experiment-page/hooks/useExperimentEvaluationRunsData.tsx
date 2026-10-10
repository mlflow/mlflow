import { useInfiniteQuery } from '@databricks/web-shared/query-client';
import type { SearchRunsApiResponse } from '@mlflow/mlflow/src/experiment-tracking/types';
import { MlflowService } from '../../../sdk/MlflowService';
import { useMemo } from 'react';

export const SEARCH_RUNS_QUERY_KEY = 'SEARCH_RUNS';

export const useExperimentEvaluationRunsData = ({
  experimentId,
  enabled,
  filter,
  fetchAllPages = false,
}: {
  experimentId: string;
  enabled: boolean;
  filter: string;
  fetchAllPages?: boolean;
}) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error } = useInfiniteQuery<
    SearchRunsApiResponse,
    Error
  >({
    queryKey: [SEARCH_RUNS_QUERY_KEY, experimentId, filter, fetchAllPages],
    queryFn: async ({ pageParam = undefined }) => {
      const requestBody = {
        experiment_ids: [experimentId],
        order_by: ['attributes.start_time DESC'],
        run_view_type: 'ACTIVE_ONLY',
        filter,
        max_results: 50,
        page_token: pageParam,
      };

      let response = await MlflowService.searchRuns(requestBody);
      if (!fetchAllPages) return response;

      const runs = [...(response.runs ?? [])];
      while (response.next_page_token) {
        response = await MlflowService.searchRuns({ ...requestBody, page_token: response.next_page_token });
        runs.push(...(response.runs ?? []));
      }
      return { ...response, runs, next_page_token: undefined };
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
  });

  const { evaluationRuns, trainingRuns } = useMemo(() => {
    if (!data?.pages) {
      return { evaluationRuns: [], trainingRuns: [] };
    }
    const allRuns = data.pages.flatMap((page) => page.runs || []);
    return allRuns.reduce(
      (acc, run) => {
        const isTrainingRun = run.outputs?.modelOutputs?.length ?? 0;

        if (isTrainingRun) {
          acc.trainingRuns.push(run);
        } else {
          acc.evaluationRuns.push(run);
        }

        return acc;
      },
      { evaluationRuns: [] as typeof allRuns, trainingRuns: [] as typeof allRuns },
    );
  }, [data]);

  return {
    data: evaluationRuns,
    trainingRuns,
    hasNextPage,
    fetchNextPage,
    refetch,
    isLoading,
    isFetching,
    error,
  };
};
