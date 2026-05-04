import { useInfiniteQuery } from '@databricks/web-shared/query-client';
import type { SearchRunsApiResponse } from '@mlflow/mlflow/src/experiment-tracking/types';
import { MlflowService } from '../../../sdk/MlflowService';
import { useMemo } from 'react';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_PYTEST } from '../../../constants';
import { EXPERIMENT_PARENT_ID_TAG } from '../utils/experimentPage.common-utils';
import type { RunEntityOrGroupData } from '../../../pages/experiment-evaluation-runs/ExperimentEvaluationRunsPage.utils';

export const useExperimentEvaluationRunsData = ({
  experimentId,
  enabled,
  filter,
}: {
  experimentId: string;
  enabled: boolean;
  filter: string;
}) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error } = useInfiniteQuery<
    SearchRunsApiResponse,
    Error
  >({
    queryKey: ['SEARCH_RUNS', experimentId, filter],
    queryFn: async ({ pageParam = undefined }) => {
      const requestBody = {
        experiment_ids: [experimentId],
        order_by: ['attributes.start_time DESC'],
        run_view_type: 'ACTIVE_ONLY',
        filter,
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
      return { evaluationRuns: [] as RunEntityOrGroupData[], trainingRuns: [] as RunEntityOrGroupData[] };
    }
    const allRuns = data.pages.flatMap((page) => page.runs || []);

    const regularEvalRuns: typeof allRuns = [];
    const trainingRunsList: typeof allRuns = [];

    for (const run of allRuns) {
      // Filter out pytest child runs — they are shown via RunViewPytestResultsTab
      const tags = run.data?.tags ?? [];
      const isPytestChild =
        tags.some((tag) => tag.key === MLFLOW_RUN_TYPE_TAG && tag.value === MLFLOW_RUN_TYPE_VALUE_PYTEST) &&
        tags.some((tag) => tag.key === EXPERIMENT_PARENT_ID_TAG);

      if (isPytestChild) {
        continue;
      }

      const isTrainingRun = run.outputs?.modelOutputs?.length ?? 0;
      if (isTrainingRun) {
        trainingRunsList.push(run);
      } else {
        regularEvalRuns.push(run);
      }
    }

    return {
      evaluationRuns: regularEvalRuns as RunEntityOrGroupData[],
      trainingRuns: trainingRunsList,
    };
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
