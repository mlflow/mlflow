import { useMemo } from 'react';
import { useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { isIssueDetectionRun } from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { MlflowService } from '../../../sdk/MlflowService';
import type { SearchRunsApiResponse } from '../../../types';

// Same split as `useExperimentEvaluationRunsData`: runs that logged a model are training runs, and
// detection runs are not listed at all, so neither counts towards the total.
const countable = (runs: SearchRunsApiResponse['runs']) =>
  (runs ?? []).filter((run) => !(run.outputs?.modelOutputs?.length ?? 0) && !isIssueDetectionRun(run));

/**
 * First page of the experiment's evaluation runs without any toolbar filter, for "Showing N of M".
 * `useQueries` with an empty list creates no query observer or cache entry, so while this is off
 * the query cache is the same as without it.
 */
export const useEvalRunsUnfilteredCount = ({ experimentId, enabled }: { experimentId: string; enabled: boolean }) => {
  const [result] = useQueries({
    queries: (enabled ? [experimentId] : []).map((id) => ({
      queryKey: ['EVAL_RUNS_UNFILTERED_COUNT', id],
      queryFn: async (): Promise<SearchRunsApiResponse> => {
        const requestBody = {
          experiment_ids: [id],
          order_by: ['attributes.start_time DESC'],
          run_view_type: 'ACTIVE_ONLY',
          filter: '',
          max_results: 50,
        };
        let response = await MlflowService.searchRuns(requestBody);
        let runs = countable(response.runs);
        // An experiment whose newest 50 runs are all detection runs would otherwise count zero,
        // which the page reads as "this experiment has no runs" and answers with first-run
        // onboarding instead of the filtered-empty state. Page on until something counts, so the
        // total stays a lower bound rather than becoming a wrong zero.
        while (runs.length === 0 && response.next_page_token) {
          response = await MlflowService.searchRuns({ ...requestBody, page_token: response.next_page_token });
          runs = countable(response.runs);
        }
        return { ...response, runs };
      },
      cacheTime: 0,
      refetchOnWindowFocus: false,
      retry: false,
    })),
  });
  const data = result?.data;
  const runs = useMemo(() => data?.runs ?? [], [data]);
  return {
    runs,
    hasNextPage: Boolean(data?.next_page_token),
    isLoading: Boolean(result?.isLoading),
    error: result?.error ?? null,
  };
};
