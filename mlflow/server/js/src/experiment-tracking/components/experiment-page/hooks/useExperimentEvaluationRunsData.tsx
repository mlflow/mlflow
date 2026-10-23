import { useInfiniteQuery } from '@databricks/web-shared/query-client';
import type { RunEntity, SearchRunsApiResponse } from '@mlflow/mlflow/src/experiment-tracking/types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../../constants';
import { MlflowService } from '../../../sdk/MlflowService';
import { useMemo } from 'react';

export const SEARCH_RUNS_QUERY_KEY = 'SEARCH_RUNS';
const DEFAULT_ORDER_BY = ['attributes.start_time DESC'];

/**
 * Issue detection runs back the Issues tab rather than recording something a user evaluated, so
 * they are kept out of the evaluation runs list. Filtered here rather than in the search filter
 * because a `!=` clause on a tag matches only runs that carry that tag, which would also drop
 * every run with no run type at all -- that is, nearly all of them.
 */
export const isIssueDetectionRun = (run: RunEntity) =>
  Boolean(
    run.data?.tags?.some(
      ({ key, value }) => key === MLFLOW_RUN_TYPE_TAG && value === MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
    ),
  );

export const useExperimentEvaluationRunsData = ({
  experimentId,
  enabled,
  filter,
  orderBy = DEFAULT_ORDER_BY,
  fetchAllPages = false,
  excludeIssueDetectionRuns = false,
}: {
  experimentId: string;
  enabled: boolean;
  filter: string;
  orderBy?: string[];
  fetchAllPages?: boolean;
  excludeIssueDetectionRuns?: boolean;
}) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error, dataUpdatedAt } = useInfiniteQuery<
    SearchRunsApiResponse,
    Error
  >({
    queryKey: [
      SEARCH_RUNS_QUERY_KEY,
      experimentId,
      filter,
      orderBy,
      fetchAllPages,
      excludeIssueDetectionRuns,
    ],
    queryFn: async ({ pageParam = undefined }) => {
      const requestBody = {
        experiment_ids: [experimentId],
        order_by: orderBy,
        run_view_type: 'ACTIVE_ONLY',
        filter,
        max_results: 50,
        page_token: pageParam,
      };

      const keep = (pageRuns: SearchRunsApiResponse['runs']) =>
        excludeIssueDetectionRuns ? (pageRuns ?? []).filter((run) => !isIssueDetectionRun(run)) : (pageRuns ?? []);

      let response = await MlflowService.searchRuns(requestBody);
      if (!fetchAllPages && !excludeIssueDetectionRuns) return response;

      const runs = [...keep(response.runs)];
      // A page made up entirely of detection runs would otherwise render as an empty list with
      // more pages waiting behind it, so keep paging until something survives the filter.
      let shouldFetchNextPage = Boolean(response.next_page_token) && (fetchAllPages || runs.length === 0);
      while (shouldFetchNextPage) {
        response = await MlflowService.searchRuns({ ...requestBody, page_token: response.next_page_token });
        runs.push(...keep(response.runs));
        shouldFetchNextPage = Boolean(response.next_page_token) && (fetchAllPages || runs.length === 0);
      }
      return { ...response, runs, next_page_token: fetchAllPages ? undefined : response.next_page_token };
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
