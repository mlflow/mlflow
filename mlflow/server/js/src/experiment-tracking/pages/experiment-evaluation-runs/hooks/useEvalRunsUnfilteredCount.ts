import { useMemo } from 'react';
import { useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { isIssueDetectionRun } from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { MlflowService } from '../../../sdk/MlflowService';
import type { SearchRunsApiResponse } from '../../../types';

/**
 * First page of the experiment's evaluation runs without any toolbar filter, for "Showing N of M".
 * `useQueries` with an empty list creates no query observer or cache entry, so while this is off
 * the query cache is the same as without it.
 */
export const useEvalRunsUnfilteredCount = ({ experimentId, enabled }: { experimentId: string; enabled: boolean }) => {
  const [result] = useQueries({
    queries: (enabled ? [experimentId] : []).map((id) => ({
      queryKey: ['EVAL_RUNS_UNFILTERED_COUNT', id],
      queryFn: (): Promise<SearchRunsApiResponse> =>
        MlflowService.searchRuns({
          experiment_ids: [id],
          order_by: ['attributes.start_time DESC'],
          run_view_type: 'ACTIVE_ONLY',
          filter: '',
          max_results: 50,
        }),
      cacheTime: 0,
      refetchOnWindowFocus: false,
      retry: false,
    })),
  });
  const data = result?.data;
  // Same split as `useExperimentEvaluationRunsData`: runs that logged a model are training runs,
  // and detection runs are not listed at all, so neither counts towards the total.
  const runs = useMemo(
    () => (data?.runs ?? []).filter((run) => !(run.outputs?.modelOutputs?.length ?? 0) && !isIssueDetectionRun(run)),
    [data],
  );
  return {
    runs,
    hasNextPage: Boolean(data?.next_page_token),
    isLoading: Boolean(result?.isLoading),
    error: result?.error ?? null,
  };
};
