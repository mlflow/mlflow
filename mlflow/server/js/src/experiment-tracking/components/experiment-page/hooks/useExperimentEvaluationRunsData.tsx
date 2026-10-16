import { useInfiniteQuery } from '@databricks/web-shared/query-client';
import type { SearchRunsApiResponse } from '@mlflow/mlflow/src/experiment-tracking/types';
import { MlflowService } from '../../../sdk/MlflowService';
import { useMemo, useRef } from 'react';

export const useExperimentEvaluationRunsData = ({
  experimentId,
  enabled,
  filter,
  resolveFilter,
}: {
  experimentId: string;
  enabled: boolean;
  filter: string;
  /**
   * Builds the filter sent with the requests, for filters with a relative date window. `filter` stays
   * the query key, so a refetch re-resolves the window instead of reusing the cutoff from when the
   * filter was set. It is called for the first page only; later pages reuse that filter, so every page
   * of one result set has the same cutoff.
   */
  resolveFilter?: () => string;
}) => {
  const resolvedFilterRef = useRef<string>();
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error, dataUpdatedAt } = useInfiniteQuery<
    SearchRunsApiResponse,
    Error
  >({
    queryKey: ['SEARCH_RUNS', experimentId, filter],
    queryFn: async ({ pageParam = undefined }) => {
      // A fetch (including a refetch) starts at the first page, where `pageParam` is undefined.
      if (resolveFilter && (pageParam === undefined || resolvedFilterRef.current === undefined)) {
        resolvedFilterRef.current = resolveFilter();
      }
      const requestBody = {
        experiment_ids: [experimentId],
        order_by: ['attributes.start_time DESC'],
        run_view_type: 'ACTIVE_ONLY',
        filter: resolveFilter ? resolvedFilterRef.current : filter,
        max_results: 50,
        page_token: pageParam,
      };

      return MlflowService.searchRuns(requestBody);
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
    dataUpdatedAt,
  };
};
